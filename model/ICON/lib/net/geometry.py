

import torch


def index(feat, uv):
    uv = uv.transpose(1, 2)

    (B, N, _) = uv.shape
    C = feat.shape[1]

    if uv.shape[-1] == 3:
        uv = uv.unsqueeze(2).unsqueeze(3)
    else:
        uv = uv.unsqueeze(2)

    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)
    return samples.view(B, C, N)


def orthogonal(points, calibrations, transforms=None):
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):

    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
