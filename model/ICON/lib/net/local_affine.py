
import torch
import torch.nn as nn
import torch.sparse as sp


class LocalAffine(nn.Module):
    def __init__(self, num_points, batch_size=1, edges=None):
        super(LocalAffine, self).__init__()
        self.A = nn.Parameter(
            torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)
        )
        self.b = nn.Parameter(
            torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(
                batch_size, num_points, 1, 1
            )
        )
        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=3)
        w1 = torch.index_select(affine_weight, dim=1, index=idx1)
        w2 = torch.index_select(affine_weight, dim=1, index=idx2)
        w_diff = (w1 - w2)**2
        w_rigid = (torch.linalg.det(self.A) - 1.0)**2
        return w_diff, w_rigid

    def forward(self, x, return_stiff=False):
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(3)
        if return_stiff:
            stiffness, rigid = self.stiffness()
            return out_x, stiffness, rigid
        else:
            return out_x
