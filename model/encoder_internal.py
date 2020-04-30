#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings
import torch
from torch import nn


class MultiDenseLayer(nn.Module):

    def __init__(self, n_dim_in, n_dim_out, n_dim_hidden, n_layer, activation_function,
                 bias: bool = True):
        """
        multi-layer dense neural network with artibrary activation function
        output = Dense(iter(Activation(Dense())))(input)

        :param n_dim_in: input dimension size
        :param n_dim_out: output dimension size
        :param n_dim_hidden: hidden layer dimension size
        :param n_layer: number of layers
        :param activation_function: activation function. e.g. torch.relu
        """
        super().__init__()

        self._n_hidden = n_layer
        self._n_dim_out = n_dim_out
        self._bias = bias
        self._lst_dense = []
        for k in range(n_layer):
            n_in = n_dim_in if k==0 else n_dim_hidden
            n_out = n_dim_out if k==(n_layer - 1) else n_dim_hidden
            self._lst_dense.append(nn.Linear(n_in, n_out, bias=bias))
        self._activation = activation_function
        self._layers = nn.ModuleList(self._lst_dense)

    def forward(self, x):

        for k, dense in enumerate(self._layers):
            if k == 0:
                h = self._activation(dense(x))
            elif k == (self._n_hidden-1):
                h = dense(h)
            else:
                h = self._activation(dense(h))

        return h


class StackedLSTMLayer(nn.Module):

    def __init__(self, n_dim_in, n_dim_out, n_dim_hidden, n_layer, n_seq_len,
                 time_distributed: bool = True, bidirectional: bool = False):
        super().__init__()

        self._n_dim_in = n_dim_in
        self._n_dim_out = n_dim_out
        self._n_dim_hidden = n_dim_hidden
        self._n_layer = n_layer
        self._n_seq_len = n_seq_len
        self._time_distributed = time_distributed

        self._lstm = nn.LSTM(input_size=n_dim_in, hidden_size=n_dim_hidden, num_layers=n_layer, bias=True, batch_first=True, bidirectional=bidirectional)
        linear_in_features = n_dim_hidden*2 if bidirectional else n_dim_hidden
        if time_distributed:
            self._linear = nn.Linear(in_features=linear_in_features, out_features=n_dim_out, bias=True)
        else:
            lst_layers = [nn.Linear(in_features=linear_in_features, out_features=n_dim_out, bias=True) for _ in range(n_seq_len)]
            self._linear = nn.ModuleList(lst_layers)

    def _init_bias_to_specific_digit(self, digit: int):
        if self._time_distributed:
            bias = self._linear.bias
            dtype, device = bias.dtype, bias.device
            init_bias_value = [0]*self._n_dim_out
            if digit == 0:
                init_bias_value[0] = 10
            elif digit == self._n_seq_len:
                init_bias_value[0] = -10
            else:
                warnings.warn(f"when `time_distributed` is enabled, you can't specify {digit}")
            bias.data = torch.tensor(init_bias_value, dtype=dtype, device=device)
        else:
            for idx, linear_layer in enumerate(self._linear):
                bias = linear_layer.bias
                dtype, device = bias.dtype, bias.device
                init_bias_value = [0]*self._n_dim_out
                if idx >= digit:
                    init_bias_value[0] = 10
                bias.data = torch.tensor(init_bias_value, dtype=dtype, device=device)

    def init_bias_to_min(self):
        warnings.warn(f"initialize code length: 0")
        self._init_bias_to_specific_digit(digit=0)

    def init_bias_to_max(self):
        warnings.warn(f"initialize code length: {self._n_seq_len}")
        self._init_bias_to_specific_digit(digit=self._n_seq_len)

    def forward(self, x: torch.Tensor):

        n_batch = x.shape[0]

        # copy input with specified size
        # x: (n_batch, n_dim_in) -> x_in: (n_batch, n_seq_len, n_dim_in)
        x_in = x.repeat((1, self._n_seq_len)).reshape((n_batch, self._n_seq_len, -1))

        # process as a sequence
        # z: (n_batch, n_seq_len, n_dim_hidden)
        z, _ = self._lstm(x_in)

        # apply linear transformation
        # y: (n_batch, n_seq_len, n_dim_out)
        if self._time_distributed:
            y = self._linear(z)
        else:
            lst_z = torch.unbind(z, dim=1)
            lst_y = [linear(z_n) for linear, z_n in zip(self._linear, lst_z)]
            y = torch.stack(lst_y, dim=1)

        return y