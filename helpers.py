#
# Created on Sun Dec 24 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Residual block for VQ-VAE.
    """

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()

        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, inputs):
        return self._block(inputs) + inputs


class ResidualStack(nn.Module):
    """Stack of residual blocks for VQ-VAE.
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers

        self._layers = nn.ModuleList([Residual(in_channels=in_channels,
                                               num_hiddens=num_hiddens,
                                               num_residual_hiddens=num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, inputs):
        for i in range(self._num_residual_layers):
            inputs = self._layers[i](inputs)
        return F.relu(inputs)
