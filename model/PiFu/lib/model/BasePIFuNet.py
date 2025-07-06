import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import index, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 error_term=nn.MSELoss(),
                 ):

        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.error_term = error_term

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None

    def forward(self, points, images, calibs, transforms=None):

        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):

        None

    def query(self, points, calibs, transforms=None, labels=None):

        None

    def get_preds(self):

        return self.preds

    def get_error(self):
 
        return self.error_term(self.preds, self.labels)
