

from lib.net.FBNet import define_G
from lib.net.net_util import init_net, VGGLoss
from lib.net.HGFilters import *
from lib.net.BasePIFuNet import BasePIFuNet
import torch
import torch.nn as nn


class NormalNet(BasePIFuNet):
    def __init__(self, cfg, error_term=nn.SmoothL1Loss()):

        super(NormalNet, self).__init__(error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()

        self.opt = cfg.net

        if self.training:
            self.vgg_loss = [VGGLoss()]

        self.in_nmlF = [
            item[0] for item in self.opt.in_nml if '_F' in item[0] or item[0] == 'image'
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml if '_B' in item[0] or item[0] == 'image'
        ]
        self.in_nmlF_dim = sum(
            [item[1] for item in self.opt.in_nml if '_F' in item[0] or item[0] == 'image']
        )
        self.in_nmlB_dim = sum(
            [item[1] for item in self.opt.in_nml if '_B' in item[0] or item[0] == 'image']
        )

        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

        init_net(self)

    def forward(self, in_tensor):

        inF_list = []
        inB_list = []

        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])
        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])

        nmlF = self.netF(torch.cat(inF_list, dim=1))
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        nmlF = nmlF / torch.norm(nmlF, dim=1, keepdim=True)
        nmlB = nmlB / torch.norm(nmlB, dim=1, keepdim=True)


        mask = (in_tensor['image'].abs().sum(dim=1, keepdim=True) != 0.0).detach().float()

        nmlF = nmlF * mask
        nmlB = nmlB * mask

        return nmlF, nmlB

    def get_norm_error(self, prd_F, prd_B, tgt):

        tgt_F, tgt_B = tgt['normal_F'], tgt['normal_B']

        l1_F_loss = self.l1_loss(prd_F, tgt_F)
        l1_B_loss = self.l1_loss(prd_B, tgt_B)

        with torch.no_grad():
            vgg_F_loss = self.vgg_loss[0](prd_F, tgt_F)
            vgg_B_loss = self.vgg_loss[0](prd_B, tgt_B)

        total_loss = [5.0 * l1_F_loss + vgg_F_loss, 5.0 * l1_B_loss + vgg_B_loss]

        return total_loss
