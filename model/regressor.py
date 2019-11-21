#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F


class OrdinalLogisticRegressionBasedCDFEstimator(nn.Module):

    def __init__(self, n_dim_input: int, n_output: int,
                 n_dim_hidden: Optional[int] = None, n_mlp_layer: Optional[int] = 3, dtype=torch.float32):

        super(OrdinalLogisticRegressionBasedCDFEstimator, self).__init__()

        self._n_dim_input = n_dim_input
        self._n_output = n_output
        self._n_dim_hidden = int(n_output * n_dim_input // 2) if n_dim_hidden is None else n_dim_hidden
        self._n_mlp_layer = n_mlp_layer
        self._dtype = dtype

        self._build()

    def _build(self):

        # linear transformation layers
        lst_mlp_layer = []
        self.activation_function = F.relu
        for idx in range(self._n_mlp_layer):
            n_in = self._n_dim_input if idx == 0 else self._n_dim_hidden
            n_out = 1 if idx == (self._n_mlp_layer - 1) else self._n_dim_hidden
            layer = nn.Linear(in_features=n_in, out_features=n_out)
            lst_mlp_layer.append(layer)
        self.lst_mlp_layer = nn.ModuleList(lst_mlp_layer)

        # logistic regression layers
        _base_intercepts = torch.ones(size=(1,self._n_output), dtype=self._dtype, requires_grad=True)
        _intercepts = torch.cumsum(F.softplus(_base_intercepts), dim=-1)
        self.intercepts = nn.Parameter(_intercepts)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:

        # t_z: (N_b, N_in) -> (N_b, 1)
        t_z = input_x
        for idx, mlp_layer in enumerate(self.lst_mlp_layer):
            if idx != (self._n_mlp_layer - 1):
                t_z = self.activation_function(mlp_layer.forward(t_z))
            else:
                t_z = mlp_layer.forward(t_z)

        # probs: (N_b, N_out)
        probs = F.sigmoid(t_z + self.intercepts)

        return probs


class SoftmaxBasedCDFEstimator(nn.Module):

    def __init__(self, n_dim_input: int, n_output: int,
                 n_dim_hidden: Optional[int] = None, n_mlp_layer: Optional[int] = 3, dtype=torch.float32):

        super(SoftmaxBasedCDFEstimator, self).__init__()

        self._n_dim_input = n_dim_input
        self._n_output = n_output
        self._n_dim_hidden = int(n_output * n_dim_input // 2) if n_dim_hidden is None else n_dim_hidden
        self._n_mlp_layer = n_mlp_layer
        self._dtype = dtype

        self._build()

    def _build(self):

        # linear transformation layers
        lst_mlp_layer = []
        self.activation_function = F.tanh
        for idx in range(self._n_mlp_layer):
            n_in = self._n_dim_input if idx == 0 else self._n_dim_hidden
            n_out = self._n_output + 1 if idx == (self._n_mlp_layer - 1) else self._n_dim_hidden
            layer = nn.Linear(in_features=n_in, out_features=n_out)
            lst_mlp_layer.append(layer)
        self.lst_mlp_layer = nn.ModuleList(lst_mlp_layer)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:

        # t_z: (N_b, N_in) -> (N_b, N_out+1)
        t_z = input_x
        for idx, mlp_layer in enumerate(self.lst_mlp_layer):
            if idx != (self._n_mlp_layer - 1):
                t_z = self.activation_function(mlp_layer.forward(t_z))
            else:
                t_z = mlp_layer.forward(t_z)

        # probs: (N_b, N_out)
        probs = torch.cumsum(F.softmax(t_z, dim=-1), dim=-1)[:,:self._n_output]

        return probs

