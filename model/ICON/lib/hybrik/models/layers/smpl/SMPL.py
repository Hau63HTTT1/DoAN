from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from .lbs import lbs, hybrik, rotmat_to_quat, quat_to_rotmat, rotation_matrix_to_angle_axis

try:
    import cPickle as pk
except ImportError:
    import pickle as pk

ModelOutput = namedtuple('ModelOutput', ['vertices', 'joints', 'joints_from_verts', 'rot_mats'])
ModelOutput.__new__.__defaults__ = (None, ) * len(ModelOutput._fields)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',    
        'spine1',
        'left_knee',
        'right_knee', 
        'spine2',
        'left_ankle',
        'right_ankle', 
        'spine3',
        'left_foot',
        'right_foot',  
        'neck',
        'left_collar',
        'right_collar',  
        'jaw',   
        'left_shoulder',
        'right_shoulder',   
        'left_elbow',
        'right_elbow', 
        'left_wrist',
        'right_wrist',  
        'left_thumb',
        'right_thumb',   
        'head',
        'left_middle',
        'right_middle',  
        'left_bigtoe',
        'right_bigtoe' 
    ]
    LEAF_NAMES = ['head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe']
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(
        self, model_path, h36m_jregressor, gender='neutral', dtype=torch.float32, num_joints=29
    ):
        super(SMPL_layer, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f
        self.register_buffer(
            'faces_tensor', to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long)
        )

        self.register_buffer('v_template', to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        self.register_buffer('shapedirs', to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))

        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))

        self.register_buffer(
            'J_regressor', to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype)
        )
        self.register_buffer('J_regressor_h36m', to_tensor(to_np(h36m_jregressor), dtype=dtype))

        self.num_joints = num_joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]

        self.register_buffer('children_map', self._parents_to_children(parents))
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights', to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self, pose_axis_angle, betas, global_orient, transl=None, return_verts=True):
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle
        pose2rot = True
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(
            betas,
            full_pose,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.J_regressor_h36m,
            self.parents,
            self.lbs_weights,
            pose2rot=pose2rot,
            dtype=self.dtype
        )

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - \
                joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(
                    1).detach()
            joints = joints - \
                joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - \
                joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(
                    1).detach()

        output = ModelOutput(
            vertices=vertices,
            joints=joints,
            rot_mats=rot_mats,
            joints_from_verts=joints_from_verts_h36m
        )
        return output

    def hybrik(
        self,
        pose_skeleton,
        betas,
        phis,
        global_orient,
        transl=None,
        return_verts=True,
        leaf_thetas=None
    ):
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        vertices, new_joints, rot_mats, joints_from_verts = hybrik(
            betas,
            global_orient,
            pose_skeleton,
            phis,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.J_regressor_h36m,
            self.parents,
            self.children_map,
            self.lbs_weights,
            dtype=self.dtype,
            train=self.training,
            leaf_thetas=leaf_thetas
        )

        rot_mats = rot_mats.reshape(batch_size, 24, 3, 3)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - \
                joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - \
                new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
           

        output = ModelOutput(
            vertices=vertices,
            joints=new_joints,
            rot_mats=rot_mats,
            joints_from_verts=joints_from_verts
        )
        return output
