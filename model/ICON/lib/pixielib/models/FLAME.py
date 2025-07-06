
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F


class FLAMETex(nn.Module):

    def __init__(self, config):
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.flame_tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.
        else:
            print('texture type ', config.tex_type, 'not exist!')
            raise NotImplementedError

        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode=None):
        '''
        texcode: [batchsize, n_tex]
        texture: [bz, 3, 256, 256], range: 0-1
        '''
        texture = self.texture_mean + \
            (self.texture_basis*texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


def texture_flame2smplx(cached_data, flame_texture, smplx_texture):

    if smplx_texture.shape[0] != smplx_texture.shape[1]:
        print('SMPL-X texture not squared (%d != %d)' % (smplx_texture[0], smplx_texture[1]))
        return
    if smplx_texture.shape[0] != cached_data['target_resolution']:
        print(
            'SMPL-X texture size does not match cached image resolution (%d != %d)' %
            (smplx_texture.shape[0], cached_data['target_resolution'])
        )
        return
    x_coords = cached_data['x_coords']
    y_coords = cached_data['y_coords']
    target_pixel_ids = cached_data['target_pixel_ids']
    source_uv_points = cached_data['source_uv_points']

    source_tex_coords = np.zeros_like((source_uv_points)).astype(int)
    source_tex_coords[:, 0] = np.clip(
        flame_texture.shape[0] * (1.0 - source_uv_points[:, 1]), 0.0, flame_texture.shape[0]
    ).astype(int)
    source_tex_coords[:, 1] = np.clip(
        flame_texture.shape[1] * (source_uv_points[:, 0]), 0.0, flame_texture.shape[1]
    ).astype(int)

    smplx_texture[y_coords[target_pixel_ids].astype(int),
                  x_coords[target_pixel_ids].astype(int), :] = flame_texture[source_tex_coords[:,
                                                                                               0],
                                                                             source_tex_coords[:,
                                                                                               1]]

    return smplx_texture
