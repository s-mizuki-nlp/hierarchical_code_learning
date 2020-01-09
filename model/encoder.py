#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings
from typing import List, Optional
import inspect
import torch
from torch import nn
from torch.nn import functional as F
from .regressor import SoftmaxBasedCDFEstimator, ScheduledSoftmaxBasedCDFEstimator
from .functional import CodeProbabiltyAdjuster
from .encoder_internal import MultiDenseLayer, StackedLSTMLayer


class SimpleEncoder(nn.Module):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 dtype=torch.float32, **kwargs):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype

        self._build()

    def _build(self):

        self.x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)
        self.lst_h_to_z = nn.ModuleList([nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary) for n in range(self._n_digits)])

    def forward(self, input_x: torch.Tensor):

        t_h = torch.tanh(self.x_to_h(input_x))
        lst_z = [torch.log(F.softplus(h_to_z(t_h))) for h_to_z in self.lst_h_to_z]
        lst_prob_c = [F.softmax(t_z, dim=1) for t_z in lst_z]

        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c


class CodeLengthAwareEncoder(SimpleEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 internal_layer_class: Optional[nn.Module] = None,
                 code_length_predictor_class: Optional[nn.Module] = None,
                 dtype=torch.float32,
                 **kwargs_for_code_length_predictor):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype
        self._code_length_predictor_class = code_length_predictor_class
        self._internal_layer_class = internal_layer_class
        self._n_ary_internal = None

        self._build(**kwargs_for_code_length_predictor)

    def _build(self, **kwargs_for_code_length_predictor):

        # x -> h: h = tanh(W*x+b)
        self.x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)

        # if adjuster class is specified, p(c_n) -> adjust so that p(c_n-1=0) <= p(c_n=0)
        # if Estimator class is specified, x -> p(c_n=0): p(c_n=0) = \sum_{d=1 to n} g_d(x)
        if (self._code_length_predictor_class is None) or \
            (CodeProbabiltyAdjuster in inspect.getmro(self._code_length_predictor_class)):
            self.code_length_predictor = CodeProbabiltyAdjuster.apply
            self._n_ary_internal = self._n_ary
        elif SoftmaxBasedCDFEstimator in inspect.getmro(self._code_length_predictor_class):
            self.code_length_predictor = SoftmaxBasedCDFEstimator(n_dim_input=self._n_dim_emb, n_output=self._n_digits,
                                                                  dtype=self._dtype,
                                                                  **kwargs_for_code_length_predictor)
            self._n_ary_internal = self._n_ary - 1
        elif ScheduledSoftmaxBasedCDFEstimator in inspect.getmro(self._code_length_predictor_class):
            self.code_length_predictor = ScheduledSoftmaxBasedCDFEstimator(n_dim_input=self._n_dim_emb, n_output=self._n_digits,
                                                                           dtype=self._dtype,
                                                                           **kwargs_for_code_length_predictor)
            self._n_ary_internal = self._n_ary - 1
        else:
            raise NotImplementedError(f"unsupported layer was specified: {self._code_length_predictor_class.__class__}")

        # h -> z_n: z_n = softplus(f_n(h))
        n_dim_h, n_dim_z = self._n_dim_hidden, self._n_ary_internal
        if (self._internal_layer_class is None) or (nn.Linear in inspect.getmro(self._internal_layer_class)):
            lst_layers = [nn.Linear(in_features=n_dim_h, out_features=n_dim_z) for _ in range(self._n_digits)]
        elif MultiDenseLayer in inspect.getmro(self._internal_layer_class):
            lst_layers = []
            for _ in range(self._n_digits):
                l = MultiDenseLayer(n_dim_in=n_dim_h, n_dim_out=n_dim_z, n_dim_hidden=n_dim_h, n_layer=3, activation_function=F.relu,
                                    bias=False)
                lst_layers.append(l)
        elif StackedLSTMLayer in inspect.getmro(self._internal_layer_class):
            l = StackedLSTMLayer(n_dim_in=n_dim_h, n_dim_out=n_dim_z, n_dim_hidden=n_dim_h, n_layer=1, n_seq_len=self._n_digits)
            if self._n_ary_internal == self._n_ary:
                init_code_length = kwargs_for_code_length_predictor.get("init_code_length", None)
                if init_code_length is None:
                    pass
                elif init_code_length == "min":
                    l.init_bias_to_min()
                elif init_code_length == "max":
                    l.init_bias_to_max()
            lst_layers = [l]
        else:
            raise NotImplementedError(f"unsupported layer was specified: {self._internal_layer_class.__class__}")
        self.lst_h_to_z = nn.ModuleList(lst_layers)

    def forward(self, input_x: torch.Tensor):

        # code probability: p(c_n=m)
        t_h = torch.tanh(self.x_to_h(input_x))
        if StackedLSTMLayer in inspect.getmro(self._internal_layer_class):
            h_to_z = self.lst_h_to_z[0]
            # t_z: (n_batch, n_digits, n_ary_internal)
            t_z = torch.log(F.softplus(h_to_z(t_h)) + 1E-6)
            # lst_z: [(n_batch, n_ary_internal)]*n_digits
            lst_z = list(map(torch.squeeze, t_z.split(1, dim=1)))
        else:
            lst_z = [torch.log(F.softplus(h_to_z(t_h)) + 1E-6) for h_to_z in self.lst_h_to_z]
        lst_prob_c = [F.softmax(t_z, dim=-1) for t_z in lst_z]

        # t_prob_c: (n_batch, n_digits, n_ary_internal)
        t_prob_c = torch.stack(lst_prob_c, dim=1)

        if self._n_ary_internal == self._n_ary - 1:
            # predict zero probability: p(c_n=0) separately using code length predictor
            # t_prob_c_zero: (n_batch, n_digits) -> (n_batch, n_digits, 1)
            t_prob_c_zero = self.code_length_predictor.forward(input_x)
            t_prob_c_zero = t_prob_c_zero.unsqueeze(-1)

            # concat it with code probability
            # t_prob_c: (n_batch, n_digits, n_ary)
            t_prob_c = torch.cat((t_prob_c_zero, (1.0-t_prob_c_zero)*t_prob_c), dim=-1)
        elif self._n_ary_internal == self._n_ary:
            # otherwise, simply adjust code probability so that it satisfies monotone increasing condition.
            t_prob_c = self.code_length_predictor(t_prob_c)
        else:
            raise ValueError(f"unexpected internal value: {self._n_ary_internal}")

        return t_prob_c

    @property
    def gate_open_ratio(self):
        return getattr(self.code_length_predictor, "gate_open_ratio", None)

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        if self.gate_open_ratio is not None:
            setattr(self.code_length_predictor, "gate_open_ratio", value)