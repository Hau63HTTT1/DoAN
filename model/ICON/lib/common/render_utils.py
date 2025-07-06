
import torch
from torch import nn
import trimesh
import math
from typing import NewType
from pytorch3d.structures import Meshes 
from pytorch3d.renderer.mesh import rasterize_meshes

Tensor = NewType('Tensor', torch.Tensor)


def solid_angles(points: Tensor, triangles: Tensor, thresh: float = 1e-8) -> Tensor:    
    centered_tris = triangles[:, None] - points[:, :, None, None]

    
    norms = torch.norm(centered_tris, dim=-1)

    cross_prod = torch.cross(centered_tris[:, :, :, 1], centered_tris[:, :, :, 2], dim=-1)
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (
        norms.prod(dim=-1) + dot01 * norms[:, :, :, 2] + dot02 * norms[:, :, :, 1] +
        dot12 * norms[:, :, :, 0]
    )
    del dot01, dot12, dot02, norms

    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle


def winding_numbers(points: Tensor, triangles: Tensor, thresh: float = 1e-8) -> Tensor:
    return 1 / (4 * math.pi) * solid_angles(points, triangles, thresh=thresh).sum(dim=-1)


def batch_contains(verts, faces, points):

    B = verts.shape[0]
    N = points.shape[1]

    verts = verts.detach().cpu()
    faces = faces.detach().cpu()
    points = points.detach().cpu()
    contains = torch.zeros(B, N)

    for i in range(B):
        contains[i] = torch.as_tensor(trimesh.Trimesh(verts[i], faces[i]).contains(points[i]))

    return 2.0 * (contains - 0.5)


def dict2obj(d):
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def face_vertices(vertices, faces):
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


class Pytorch3dRasterizer(nn.Module):

    def __init__(self, image_size=224):

        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': -1,
            'max_faces_per_bin': None,
            'perspective_correct': True,
            'cull_backfaces': True,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(
            attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1]
        )
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0    
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals
