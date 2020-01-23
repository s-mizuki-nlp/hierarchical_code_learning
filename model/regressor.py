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
import numpy as np


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
                 n_dim_hidden: Optional[int] = None, n_mlp_layer: Optional[int] = 3,
                 assign_nonzero_value_on_most_significant_digit: bool = False,
                 init_code_length: Optional[str] = None,
                 dtype=torch.float32):

        super(SoftmaxBasedCDFEstimator, self).__init__()

        self._n_dim_input = n_dim_input
        self._n_output = n_output
        if assign_nonzero_value_on_most_significant_digit:
            self._n_output_softmax = n_output
        else:
            self._n_output_softmax = n_output + 1

        self._n_dim_hidden = int(n_output * n_dim_input // 2) if n_dim_hidden is None else n_dim_hidden
        self._n_mlp_layer = n_mlp_layer
        self._msd_nonzero = assign_nonzero_value_on_most_significant_digit
        self._dtype = dtype

        self._build()

        if init_code_length is not None:
            if init_code_length == "min":
                self._init_bias_to_min()
            elif init_code_length == "max":
                self._init_bias_to_max()
            elif init_code_length == "random":
                self._init_weight_and_bias_to_random()
            else:
                raise NotImplementedError(f"unknown value was specified: {init_code_length}")

    def _build(self):

        # linear transformation layers
        lst_mlp_layer = []
        self.activation_function = torch.tanh

        for idx in range(self._n_mlp_layer):
            n_in = self._n_dim_input if idx == 0 else self._n_dim_hidden
            n_out = self._n_output_softmax if idx == (self._n_mlp_layer - 1) else self._n_dim_hidden
            layer = nn.Linear(in_features=n_in, out_features=n_out)
            lst_mlp_layer.append(layer)
        self.lst_mlp_layer = nn.ModuleList(lst_mlp_layer)

    def _init_bias_to_min(self):
        final_layer_bias = self.lst_mlp_layer[-1].bias
        dtype, device = final_layer_bias.dtype, final_layer_bias.device
        final_layer_bias.data = torch.tensor((10,) + (0,)*(self._n_output_softmax-1), dtype=dtype, device=device)

    def _init_bias_to_max(self):
        final_layer_bias = self.lst_mlp_layer[-1].bias
        dtype, device = final_layer_bias.dtype, final_layer_bias.device
        final_layer_bias.data = torch.tensor((0,)*(self._n_output_softmax-1) + (10,), dtype=dtype, device=device)

    def _init_weight_and_bias_to_random(self):
        final_layer_weight = self.lst_mlp_layer[-1].weight
        final_layer_bias = self.lst_mlp_layer[-1].bias
        dtype, device = final_layer_bias.dtype, final_layer_bias.device
        final_layer_bias.data = torch.tensor((0,)*self._n_output_softmax, dtype=dtype, device=device)
        final_layer_weight.data *= 3 * np.sqrt(self._n_output_softmax)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:

        # t_z: (N_b, N_in) -> (N_b, N_out+1)
        t_z = input_x
        for idx, mlp_layer in enumerate(self.lst_mlp_layer):
            if idx != (self._n_mlp_layer - 1):
                t_z = self.activation_function(mlp_layer.forward(t_z))
            else:
                t_z = mlp_layer.forward(t_z)

        # probs: (N_b, N_out)
        probs = torch.cumsum(F.softmax(t_z, dim=-1), dim=-1)[:,:(self._n_output_softmax - 1)]
        if self._msd_nonzero:
            device = input_x.device
            n_batch = input_x.shape[0]
            pad_tensor = torch.ones((n_batch, 1), dtype=self._dtype, device=device) * 1E-6
            probs = torch.cat((pad_tensor, probs), dim=-1)

        return probs


class ScheduledSoftmaxBasedCDFEstimator(SoftmaxBasedCDFEstimator):

    _EPS = 0.01

    def __init__(self, n_dim_input: int, n_output: int,
                 n_dim_hidden: Optional[int] = None, n_mlp_layer: Optional[int] = 3,
                 assign_nonzero_value_on_most_significant_digit: bool = True,
                 init_code_length: Optional[str] = None,
                 dtype=torch.float32):

        super().__init__(n_dim_input, n_output, n_dim_hidden, n_mlp_layer, assign_nonzero_value_on_most_significant_digit,
                         init_code_length, dtype)
        self._gate_open_ratio = 0.0

        self._offset = -0.5 if assign_nonzero_value_on_most_significant_digit else 0.5
        self._coef_gamma = 2 * np.log((1. - self._EPS) / self._EPS)
        self._coef_alpha = float(n_output - 1 if assign_nonzero_value_on_most_significant_digit else n_output)

    @property
    def gate_open_ratio(self) -> float:
        return self._gate_open_ratio

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        self._gate_open_ratio = value

    def _calc_gate_mask(self, device):
        # gate close <=> gate_mask=1, gate open <=> gate_mask=0
        # gate full close <=> gate_open_ratio=0, gate full open <=> gate_open_ratio=1
        # formula: gate_mask[n] = \sigma(\gamma*(n + offset - \alpha * r)
        # it always holds: gate_mask[n+1] > gate_mask[n]
        # when r=0 (=full close), gate_mask[0] = \sigma(\gamma*offset) = \epsilon if msd_nonzero else 1 - \epsilon
        # when r=1 (=full open), gate_mask[n_output-1] = \sigma(\gamma*(n_output-1 + offset - \alpha) = \epsilon
        vec_n = torch.arange(self._n_output, dtype=self._dtype, device=device)
        vec_intercepts = self._coef_gamma*(vec_n + self._offset - self._coef_alpha * self._gate_open_ratio)
        gate_mask = torch.sigmoid(vec_intercepts)
        return gate_mask

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:

        # probs_orig = (n_batch, n_output)
        probs_orig = super().forward(input_x)
        # gate_mask = (1, n_output)
        gate_mask = self._calc_gate_mask(device=input_x.device).unsqueeze(dim=0)

        # apply gate mask by element-wise max
        probs = torch.max(gate_mask, probs_orig)

        return probs