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
from contextlib import ExitStack
from model.encoder import SimpleEncoder, CodeLengthAwareEncoder

class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):

        super(AutoEncoder, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, t_x: torch.Tensor, requires_grad: bool = True):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            # encoder
            if isinstance(self._encoder, SimpleEncoder):
                t_intermediate = self._encoder.forward(t_x)
                t_code_prob = None
            else:
                raise NotImplementedError(f"not implemented yet: {self._encoder.__class__.__name__}")

            # decoder
            t_x_dash = self._decoder.forward(t_intermediate)

        return t_intermediate, t_code_prob, t_x_dash