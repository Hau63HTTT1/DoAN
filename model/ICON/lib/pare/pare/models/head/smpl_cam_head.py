

import torch
import torch.nn as nn

from ...core import config
from .smpl_head import SMPL


class SMPLCamHead(nn.Module):
    def __init__(self, img_res=224):
        super(SMPLCamHead, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.add_module('smpl', self.smpl)

        self.img_res = img_res

    def forward(
        self,
        rotmat,
        shape,
        cam,
        cam_rotmat,
        cam_intrinsics,
        bbox_scale,
        bbox_center,
        img_w,
        img_h,
        normalize_joints2d=False
    ):
        smpl_output = self.smpl(
            betas=shape,
            body_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        output = {
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }

        joints3d = smpl_output.joints

        cam_t = convert_pare_to_full_img_cam(
            pare_cam=cam,
            bbox_height=bbox_scale * 200.,
            bbox_center=bbox_center,
            img_w=img_w,
            img_h=img_h,
            focal_length=cam_intrinsics[:, 0, 0],
            crop_res=self.img_res,
        )

        joints2d = perspective_projection(
            joints3d,
            rotation=cam_rotmat,
            translation=cam_t,
            cam_intrinsics=cam_intrinsics,
        )


        if normalize_joints2d:
            joints2d = joints2d / (self.img_res / 2.)

        output['smpl_joints2d'] = joints2d
        output['pred_cam_t'] = cam_t

        return output


def perspective_projection(points, rotation, translation, cam_intrinsics):

    K = cam_intrinsics


    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)


    projected_points = points / points[:, :, -1].unsqueeze(-1)


    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())

    return projected_points[:, :, :-1]


def convert_pare_to_full_img_cam(
    pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length, crop_res=224
):
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t
