
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F



class TempSoftmaxFusion(nn.Module):
    def __init__(self, channels=[2048 * 2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(TempSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature

        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))

    def forward(self, x, y, work=True):

        if work:

            f_in = torch.cat([x, y], dim=1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_temp = self.layers(f_in)
            f_weight = F.softmax(f_temp * self.temperature, dim=1)


            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:, [0]] * x + f_weight[:, [1]] * y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight




class GumbelSoftmaxFusion(nn.Module):
    def __init__(self, channels=[2048 * 2, 1024, 1], detach_inputs=False, detach_feature=False):
        super(GumbelSoftmaxFusion, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature


        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l + 1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)

    def forward(self, x, y, work=True):

        if work:

            f_in = torch.cat([x, y], dim=-1)
            if self.detach_inputs:
                f_in = f_in.detach()
            f_weight = self.layers(f_in)

            f_weight = f_weight - f_weight.detach() + f_weight.gt(0.5)

     
            if self.detach_feature:
                x = x.detach()
                y = y.detach()
            f_out = f_weight[:, [0]] * x + f_weight[:, [1]] * y
            x_out = f_out
            y_out = f_out
        else:
            x_out = x
            y_out = y
            f_weight = None
        return x_out, y_out, f_weight
