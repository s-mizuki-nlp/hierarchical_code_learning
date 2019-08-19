#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import ExitStack
from model.encoder import SimpleEncoder, CodeLengthAwareEncoder
from sklearn.preprocessing import OneHotEncoder

class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, dtype=torch.float32):

        super(AutoEncoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._dtype = dtype

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).type(self._dtype)

    def forward(self, t_x: torch.Tensor, requires_grad: bool = True, enable_gumbel_softmax: bool = True):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            # encoder
            t_intermediate, t_code_prob = self._encoder.forward(t_x)

            # decoder
            if enable_gumbel_softmax:
                t_x_dash = self._decoder.forward(t_intermediate)
            else:
                t_x_dash = self._decoder.forward(t_code_prob)

        return t_intermediate, t_code_prob, t_x_dash

    def _predict(self, t_x: torch.Tensor):

        return self.forward(t_x, requires_grad=False, enable_gumbel_softmax=False)

    def predict(self, mat_x: np.ndarray):

        t_x = self._numpy_to_tensor(mat_x)
        _, _, t_x_dash = self._predict(t_x)

        return t_x_dash.cpu().numpy()

    def _encode(self, t_x: torch.Tensor):

        _, t_code_prob = self._encoder.forward(t_x)
        t_code = torch.argmax(t_code_prob, dim=2, keepdim=False)
        return t_code

    def encode(self, mat_x: np.ndarray):

        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            t_x = self._numpy_to_tensor(mat_x)
            t_code = self._encode(t_x)

        return t_code.cpu().numpy()

    def _decode(self, t_code_prob: torch.Tensor):

        # t_code_prob: (N_batch, N_digits, N_ary), t_c_[b,n,m] \in [0,1]
        return self._decoder.forward(t_code_prob)

    def decode(self, mat_code: np.ndarray):

        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            n_ary = self._decoder.n_ary
            # one-hot encoding
            t_code_prob = self._numpy_to_tensor(np.eye(n_ary)[mat_code])
            t_x_dash = self._decode(t_code_prob)

        return t_x_dash.cpu().numpy()