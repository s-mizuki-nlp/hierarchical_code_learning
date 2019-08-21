#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L


class ReconstructionLoss(L._Loss):

    def __init__(self):

        super(ReconstructionLoss, self).__init__()
        # sample-wise & element-wise mean
        self._mse_loss = L.MSELoss(reduction="mean")

    def forward(self, t_x_dash, t_x):
        return self._mse_loss.forward(t_x_dash, t_x)


class MutualInformationLoss(L._Loss):

    def __init__(self):

        super(MutualInformationLoss, self).__init__()

