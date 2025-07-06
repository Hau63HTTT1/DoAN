import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet


class VhullPIFuNet(BasePIFuNet):


    def __init__(self,
                 num_views,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):
        super(VhullPIFuNet, self).__init__(
            projection_mode=projection_mode,
            error_term=error_term)
        self.name = 'vhull'

        self.num_views = num_views

        self.im_feat = None

    def filter(self, images):
        if images.shape[1] > 3:
            self.im_feat = images[:, 3:4, :, :]
        else:
            self.im_feat = images[:, 0:1, :, :]

    def query(self, points, calibs, transforms=None, labels=None):
        if labels is not None:
            self.labels = labels

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        point_local_feat = self.index(self.im_feat, xy)
        local_shape = point_local_feat.shape
        point_feat = point_local_feat.view(
            local_shape[0] // self.num_views,
            local_shape[1] * self.num_views,
            -1)
        pred = torch.prod(point_feat, dim=1)

        self.preds = pred.unsqueeze(1)
