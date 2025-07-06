# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F 

from ..geometry import index, orthogonal, perspective

class BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 ):
 
        super(BasePIFuNet, self).__init__()
        self.name = 'base'

        self.criteria = criteria

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective

        self.preds = None
        self.labels = None
        self.nmls = None
        self.labels_nml = None
        self.preds_surface = None

    def forward(self, points, images, calibs, transforms=None):

        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()

    def filter(self, images):

        None
    
    def query(self, points, calibs, trasnforms=None, labels=None):

        None

    def calc_normal(self, points, calibs, transforms=None, delta=0.1):

        None

    def get_preds(self):
  
        return self.preds

    def get_error(self, gamma=None):
  
        return self.error_term(self.preds, self.labels, gamma)

    