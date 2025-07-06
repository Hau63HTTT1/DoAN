
from .seg3d_utils import (
    create_grid3D,
    plot_mask3D,
    SmoothConv3D,
)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import mcubes
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
import logging

logging.getLogger("lightning").setLevel(logging.ERROR)


class Seg3dLossless(nn.Module):
    def __init__(
        self,
        query_func,
        b_min,
        b_max,
        resolutions,
        channels=1,
        balance_value=0.5,
        align_corners=False,
        visualize=False,
        debug=False,
        use_cuda_impl=False,
        faster=False,
        use_shadow=False,
        **kwargs
    ):
        
        super().__init__()
        self.query_func = query_func
        self.register_buffer('b_min', torch.tensor(b_min).float().unsqueeze(1))
        self.register_buffer('b_max', torch.tensor(b_max).float().unsqueeze(1))

        if type(resolutions[0]) is int:
            resolutions = torch.tensor([(res, res, res) for res in resolutions])
        else:
            resolutions = torch.tensor(resolutions)
        self.register_buffer('resolutions', resolutions)
        self.batchsize = self.b_min.size(0)
        assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels
        assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize
        self.debug = debug
        self.use_cuda_impl = use_cuda_impl
        self.faster = faster
        self.use_shadow = use_shadow

        for resolution in resolutions:
            assert resolution[0] % 2 == 1 and resolution[1] % 2 == 1, \
                f"resolution {resolution} need to be odd becuase of align_corner."

        init_coords = create_grid3D(0, resolutions[-1] - 1, steps=resolutions[0])
        init_coords = init_coords.unsqueeze(0).repeat(self.batchsize, 1, 1)
        self.register_buffer('init_coords', init_coords)

        calculated = torch.zeros(
            (self.resolutions[-1][2], self.resolutions[-1][1], self.resolutions[-1][0]),
            dtype=torch.bool
        )
        self.register_buffer('calculated', calculated)

        gird8_offsets = torch.stack(
            torch.meshgrid(
                [torch.tensor([-1, 0, 1]),
                 torch.tensor([-1, 0, 1]),
                 torch.tensor([-1, 0, 1])]
            )
        ).int().view(3, -1).t()
        self.register_buffer('gird8_offsets', gird8_offsets)

        self.smooth_conv3x3 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=3)
        self.smooth_conv5x5 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=5)
        self.smooth_conv7x7 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=7)
        self.smooth_conv9x9 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=9)

    def batch_eval(self, coords, **kwargs):
        
        coords = coords.detach()
        
        if self.align_corners:
            coords2D = coords.float() / (self.resolutions[-1] - 1)
        else:
            step = 1.0 / self.resolutions[-1].float()
            coords2D = coords.float() / self.resolutions[-1] + step / 2
        coords2D = coords2D * (self.b_max - self.b_min) + self.b_min
        
        occupancys = self.query_func(**kwargs, points=coords2D)
        if type(occupancys) is list:
            occupancys = torch.stack(occupancys)
        assert len(occupancys.size()) == 3, \
            "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def forward(self, **kwargs):
        if self.faster:
            return self._forward_faster(**kwargs)
        else:
            return self._forward(**kwargs)

    def _forward_faster(self, **kwargs):
        
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]

        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone()
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)
                if (occupancys > 0.5).sum() == 0:
                    return None

                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)

                with torch.no_grad():
                    coords_accum = coords / stride

            elif torch.equal(resolution, self.resolutions[-1]):

                with torch.no_grad():
                    valid = F.interpolate(
                        (occupancys > self.balance_value).float(),
                        size=(D, H, W),
                        mode="trilinear",
                        align_corners=True
                    )

                occupancys = F.interpolate(
                    occupancys.float(), size=(D, H, W), mode="trilinear", align_corners=True
                )

                is_boundary = valid == 0.5

            else:
                coords_accum *= 2

                with torch.no_grad():
                    valid = F.interpolate(
                        (occupancys > self.balance_value).float(),
                        size=(D, H, W),
                        mode="trilinear",
                        align_corners=True
                    )

                occupancys = F.interpolate(
                    occupancys.float(), size=(D, H, W), mode="trilinear", align_corners=True
                )

                is_boundary = (valid > 0.0) & (valid < 1.0)

                with torch.no_grad():
                    if torch.equal(resolution, self.resolutions[1]):
                        is_boundary = (self.smooth_conv9x9(is_boundary.float()) > 0)[0, 0]
                    elif torch.equal(resolution, self.resolutions[2]):
                        is_boundary = (self.smooth_conv7x7(is_boundary.float()) > 0)[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]

                    coords_accum = coords_accum.long()
                    is_boundary[coords_accum[0, :, 2], coords_accum[0, :, 1],
                                coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero(as_tuple=False).unsqueeze(0)
                    point_indices = (
                        point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W +
                        point_coords[:, :, 0]
                    )

                    R, C, D, H, W = occupancys.shape

                    # inferred value
                    coords = point_coords * stride

                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)

                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C,
                                       D * H * W).scatter_(2, point_indices,
                                                           occupancys_topk).view(R, C, D, H, W)
                )

                with torch.no_grad():
                    voxels = coords / stride
                    coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)

        return occupancys[0, 0]

    def _forward(self, **kwargs):
        
        final_W = self.resolutions[-1][0]
        final_H = self.resolutions[-1][1]
        final_D = self.resolutions[-1][2]

        calculated = self.calculated.clone()

        for resolution in self.resolutions:
            W, H, D = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            if self.visualize:
                this_stage_coords = []

            if torch.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.clone()
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(self.batchsize, self.channels, D, H, W)

                if self.visualize:
                    self.plot(occupancys, coords, final_D, final_H, final_W)

                with torch.no_grad():
                    coords_accum = coords / stride
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

            else:
                coords_accum *= 2

                with torch.no_grad():
                    valid = F.interpolate(
                        (occupancys > self.balance_value).float(),
                        size=(D, H, W),
                        mode="trilinear",
                        align_corners=True
                    )

                occupancys = F.interpolate(
                    occupancys.float(), size=(D, H, W), mode="trilinear", align_corners=True
                )

                is_boundary = (valid > 0.0) & (valid < 1.0)

                with torch.no_grad():
                  
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        depth_res = resolution[2].item()
                        depth_index = torch.linspace(0, depth_res - 1,
                                                     steps=depth_res).type_as(occupancys.device)
                        depth_index_max = torch.max(
                            (occupancys > self.balance_value) * (depth_index + 1),
                            dim=-1,
                            keepdim=True
                        )[0] - 1
                        shadow = depth_index < depth_index_max
                        is_boundary[shadow] = False
                        is_boundary = is_boundary[0, 0]
                    else:
                        is_boundary = (self.smooth_conv3x3(is_boundary.float()) > 0)[0, 0]

                    is_boundary[coords_accum[0, :, 2], coords_accum[0, :, 1],
                                coords_accum[0, :, 0]] = False
                    point_coords = is_boundary.permute(2, 1, 0).nonzero(as_tuple=False).unsqueeze(0)
                    point_indices = (
                        point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W +
                        point_coords[:, :, 0]
                    )

                    R, C, D, H, W = occupancys.shape
                    
                    occupancys_interp = torch.gather(
                        occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1)
                    )

                    
                    coords = point_coords * stride

                if coords.size(1) == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)
                if self.visualize:
                    this_stage_coords.append(coords)

                
                R, C, D, H, W = occupancys.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                occupancys = (
                    occupancys.reshape(R, C,
                                       D * H * W).scatter_(2, point_indices,
                                                           occupancys_topk).view(R, C, D, H, W)
                )

                with torch.no_grad():
                    
                    conflicts = (
                        (occupancys_interp - self.balance_value) *
                        (occupancys_topk - self.balance_value) < 0
                    )[0, 0]

                    if self.visualize:
                        self.plot(occupancys, coords, final_D, final_H, final_W)

                    voxels = coords / stride
                    coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

                while conflicts.sum() > 0:
                    if self.use_shadow and torch.equal(resolution, self.resolutions[-1]):
                        break

                    with torch.no_grad():
                        conflicts_coords = coords[0, conflicts, :]

                        if self.debug:
                            self.plot(
                                occupancys,
                                conflicts_coords.unsqueeze(0),
                                final_D,
                                final_H,
                                final_W,
                                title='conflicts'
                            )

                        conflicts_boundary = (
                            conflicts_coords.int() + self.gird8_offsets.unsqueeze(1) * stride.int()
                        ).reshape(-1, 3).long().unique(dim=0)
                        conflicts_boundary[:, 0] = (
                            conflicts_boundary[:, 0].clamp(0,
                                                           calculated.size(2) - 1)
                        )
                        conflicts_boundary[:, 1] = (
                            conflicts_boundary[:, 1].clamp(0,
                                                           calculated.size(1) - 1)
                        )
                        conflicts_boundary[:, 2] = (
                            conflicts_boundary[:, 2].clamp(0,
                                                           calculated.size(0) - 1)
                        )

                        coords = conflicts_boundary[calculated[conflicts_boundary[:, 2],
                                                               conflicts_boundary[:, 1],
                                                               conflicts_boundary[:, 0]] == False]

                        if self.debug:
                            self.plot(
                                occupancys,
                                coords.unsqueeze(0),
                                final_D,
                                final_H,
                                final_W,
                                title='coords'
                            )

                        coords = coords.unsqueeze(0)
                        point_coords = coords / stride
                        point_indices = (
                            point_coords[:, :, 2] * H * W + point_coords[:, :, 1] * W +
                            point_coords[:, :, 0]
                        )

                        R, C, D, H, W = occupancys.shape
                        
                        occupancys_interp = torch.gather(
                            occupancys.reshape(R, C, D * H * W), 2, point_indices.unsqueeze(1)
                        )

                        
                        coords = point_coords * stride

                    if coords.size(1) == 0:
                        break
                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    if self.visualize:
                        this_stage_coords.append(coords)

                    with torch.no_grad():
                        
                        conflicts = (
                            (occupancys_interp - self.balance_value) *
                            (occupancys_topk - self.balance_value) < 0
                        )[0, 0]

                    
                    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                    occupancys = (
                        occupancys.reshape(R, C,
                                           D * H * W).scatter_(2, point_indices,
                                                               occupancys_topk).view(R, C, D, H, W)
                    )

                    with torch.no_grad():
                        voxels = coords / stride
                        coords_accum = torch.cat([voxels, coords_accum], dim=1).unique(dim=1)
                        calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

                if self.visualize:
                    this_stage_coords = torch.cat(this_stage_coords, dim=1)
                    self.plot(occupancys, this_stage_coords, final_D, final_H, final_W)

        return occupancys[0, 0]

    def plot(self, occupancys, coords, final_D, final_H, final_W, title='', **kwargs):
        final = F.interpolate(
            occupancys.float(),
            size=(final_D, final_H, final_W),
            mode="trilinear",
            align_corners=True
        )    
        x = coords[0, :, 0].to("cpu")
        y = coords[0, :, 1].to("cpu")
        z = coords[0, :, 2].to("cpu")

        plot_mask3D(final[0, 0].to("cpu"), title, (x, y, z), **kwargs)

    def find_vertices(self, sdf, direction="front"):
        
        resolution = sdf.size(2)
        if direction == "front":
            pass
        elif direction == "left":
            sdf = sdf.permute(2, 1, 0)
        elif direction == "back":
            inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
            sdf = sdf[inv_idx, :, :]
        elif direction == "right":
            inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
            sdf = sdf[:, :, inv_idx]
            sdf = sdf.permute(2, 1, 0)

        inv_idx = torch.arange(sdf.size(2) - 1, -1, -1).long()
        sdf = sdf[inv_idx, :, :]
        sdf_all = sdf.permute(2, 1, 0)

        
        grad_v = (sdf_all > 0.5) * torch.linspace(resolution, 1, steps=resolution).to(sdf.device)
        grad_c = torch.ones_like(sdf_all) * torch.linspace(0, resolution - 1,
                                                           steps=resolution).to(sdf.device)
        max_v, max_c = grad_v.max(dim=2)
        shadow = grad_c > max_c.view(resolution, resolution, 1)
        keep = (sdf_all > 0.5) & (~shadow)

        p1 = keep.nonzero(as_tuple=False).t()
        p2 = p1.clone()
        p2[2, :] = (p2[2, :] - 2).clamp(0, resolution)
        p3 = p1.clone()
        p3[1, :] = (p3[1, :] - 2).clamp(0, resolution)
        p4 = p1.clone()
        p4[0, :] = (p4[0, :] - 2).clamp(0, resolution)

        v1 = sdf_all[p1[0, :], p1[1, :], p1[2, :]]
        v2 = sdf_all[p2[0, :], p2[1, :], p2[2, :]]
        v3 = sdf_all[p3[0, :], p3[1, :], p3[2, :]]
        v4 = sdf_all[p4[0, :], p4[1, :], p4[2, :]]

        X = p1[0, :].long()
        Y = p1[1, :].long()
        Z = p2[2, :].float() * (0.5 - v1) / (v2 - v1) + \
            p1[2, :].float() * (v2 - 0.5) / (v2 - v1)
        Z = Z.clamp(0, resolution)

    
        norm_z = v2 - v1
        norm_y = v3 - v1
        norm_x = v4 - v1

        norm = torch.stack([norm_x, norm_y, norm_z], dim=1)
        norm = norm / torch.norm(norm, p=2, dim=1, keepdim=True)

        return X, Y, Z, norm

    def render_normal(self, resolution, X, Y, Z, norm):
        image = torch.ones((1, 3, resolution, resolution), dtype=torch.float32).to(norm.device)
        color = (norm + 1) / 2.0
        color = color.clamp(0, 1)
        image[0, :, Y, X] = color.t()
        return image

    def display(self, sdf):

        X, Y, Z, norm = self.find_vertices(sdf, direction="front")
        image1 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction="left")
        image2 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction="right")
        image3 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)
        X, Y, Z, norm = self.find_vertices(sdf, direction="back")
        image4 = self.render_normal(self.resolutions[-1, -1], X, Y, Z, norm)

        image = torch.cat([image1, image2, image3, image4], axis=3)
        image = image.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0

        return np.uint8(image)

    def export_mesh(self, occupancys):

        final = occupancys[1:, 1:, 1:].contiguous()

        if final.shape[0] > 256:
            occu_arr = final.detach().cpu().numpy()
            vertices, triangles = mcubes.marching_cubes(occu_arr, self.balance_value)
            verts = torch.as_tensor(vertices[:, [2, 1, 0]])
            faces = torch.as_tensor(triangles.astype(np.long), dtype=torch.long)[:, [0, 2, 1]]
        else:
            torch.cuda.empty_cache()
            vertices, triangles = voxelgrids_to_trianglemeshes(final.unsqueeze(0))
            verts = vertices[0][:, [2, 1, 0]].cpu()
            faces = triangles[0][:, [0, 2, 1]].cpu()

        return verts, faces
