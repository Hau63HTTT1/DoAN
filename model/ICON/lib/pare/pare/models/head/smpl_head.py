
import torch
import numpy as np
import torch.nn as nn

from lib.smplx import SMPL as _SMPL
from lib.smplx.utils import SMPLOutput
from lib.smplx.lbs import vertices2joints

from ...core import config, constants
from ...utils.geometry import perspective_projection, convert_weak_perspective_to_perspective


class SMPL(_SMPL):
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32)
        )
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(
            vertices=smpl_output.vertices,
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            betas=smpl_output.betas,
            full_pose=smpl_output.full_pose
        )
        return output


class SMPLHead(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(SMPLHead, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.add_module('smpl', self.smpl)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):
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
        if cam is not None:
            joints3d = smpl_output.joints
            batch_size = joints3d.shape[0]
            device = joints3d.device
            cam_t = convert_weak_perspective_to_perspective(
                cam,
                focal_length=self.focal_length,
                img_res=self.img_res,
            )
            joints2d = perspective_projection(
                joints3d,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1),
                translation=cam_t,
                focal_length=self.focal_length,
                camera_center=torch.zeros(batch_size, 2, device=device)
            )
            if normalize_joints2d:
                joints2d = joints2d / (self.img_res / 2.)

            output['smpl_joints2d'] = joints2d
            output['pred_cam_t'] = cam_t

        return output
