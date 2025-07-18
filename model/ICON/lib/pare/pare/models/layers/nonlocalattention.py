

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalAttention(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
    ):
        super(NonLocalAttention, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        batch_size, n_feats, n_joints, _ = input.shape
        input = input.squeeze(-1)
        attention = torch.matmul(input.transpose(2, 1), input)
        norm_attention = F.softmax(attention, dim=-1)

        out = torch.matmul(input, norm_attention)
        out = self.conv1x1(out)

        out = out.unsqueeze(-1) 
        return out


if __name__ == '__main__':
    nla = NonLocalAttention()

    inp = torch.rand(32, 256, 24, 1)

    out = nla(inp)
    print(out.shape)
