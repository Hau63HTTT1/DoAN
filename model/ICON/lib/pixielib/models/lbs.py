

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import yaml
import torch
import torch.nn.functional as F
from torch import nn


def rot_mat_to_euler(rot_mats):

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def find_dynamic_lmk_idx_and_bcoords(
    vertices,
    pose,
    dynamic_lmk_faces_idx,
    dynamic_lmk_b_coords,
    head_kin_chain,
    dtype=torch.float32
):


    batch_size = vertices.shape[0]
    pose = pose.detach()

    rot_mats = torch.index_select(pose, 1, head_kin_chain)

    rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0)
    for idx in range(len(head_kin_chain)):

        rel_rot_mat = torch.matmul(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                                          max=39)).to(dtype=torch.long)

    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)


    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(batch_size, -1, 3)

    lmk_faces += torch.arange(batch_size, dtype=torch.long,
                              device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(
    betas,
    pose,
    v_template,
    shapedirs,
    posedirs,
    J_regressor,
    parents,
    lbs_weights,
    pose2rot=True,
    dtype=torch.float32
):


    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    v_shaped = v_template + blend_shapes(betas, shapedirs)


    J = vertices2joints(J_regressor, v_shaped)

 
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)


    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def vertices2joints(J_regressor, vertices):


    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):

    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):


    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)


    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):

    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):


    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3),
                                   rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):

        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    posed_joints = transforms[:, :, :3, 3]



    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


class JointsFromVerticesSelector(nn.Module):
    def __init__(self, fname):


        super(JointsFromVerticesSelector, self).__init__()

        err_msg = ('Either pass a filename or triangle face ids, names and'
                   ' barycentrics')
        assert fname is not None or (
            face_ids is not None and bcs is not None and names is not None
        ), err_msg
        if fname is not None:
            fname = os.path.expanduser(os.path.expandvars(fname))
            with open(fname, 'r') as f:
                data = yaml.safe_load(f)
            names = list(data.keys())
            bcs = []
            face_ids = []
            for name, d in data.items():
                face_ids.append(d['face'])
                bcs.append(d['bc'])
            bcs = np.array(bcs, dtype=np.float32)
            face_ids = np.array(face_ids, dtype=np.int32)
        assert len(bcs) == len(face_ids), (
            'The number of barycentric coordinates must be equal to the faces'
        )
        assert len(names) == len(face_ids), ('The number of names must be equal to the number of ')

        self.names = names
        self.register_buffer('bcs', torch.tensor(bcs, dtype=torch.float32))
        self.register_buffer('face_ids', torch.tensor(face_ids, dtype=torch.long))

    def extra_joint_names(self):

        return self.names

    def forward(self, vertices, faces):
        if len(self.face_ids) < 1:
            return []
        vertex_ids = faces[self.face_ids].reshape(-1)

        triangles = torch.index_select(vertices, 1, vertex_ids).reshape(-1, len(self.bcs), 3, 3)
        return (triangles * self.bcs[None, :, :, None]).sum(dim=2)




def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
