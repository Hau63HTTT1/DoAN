
import torch
import torch.nn as nn
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, filter_channels, name=None, res_layers=[], norm='group', last_op=None):

        super(MLP, self).__init__()

        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.name = name
        self.activate = nn.LeakyReLU(inplace=True)

        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1)
                )
            else:
                self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm1d(filter_channels[l + 1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l], name='weight')

    def forward(self, feature):
        y = feature
        tmpy = feature

        for i, f in enumerate(self.filters):

            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))

        if self.last_op is not None:
            y = self.last_op(y)

        return y
