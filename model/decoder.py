#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F

class SimpleDecoder(nn.Module):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int, **kwargs):

        super(SimpleDecoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_digits = n_digits
        self._n_ary = n_ary

        self._build()

    @property
    def n_ary(self):
        return self._n_ary

    def _build(self):
        self.lst_c_to_x = nn.ModuleList([nn.Linear(in_features=self._n_ary, out_features=self._n_dim_emb, bias=False) for d in range(self._n_digits)])

    def forward(self, input_c: torch.Tensor):

        # input_c: (N_b, N_digits, N_ary)
        lst_x = []
        for d in range(self._n_digits):
            # lst_x[n]: (N_b, N_dim_emb)
            lst_x.append(self.lst_c_to_x[d](input_c[:,d,:]))

        # t_x_dash: \sum_n{(N_b, n, N_dim_emb)} = (N_b, N_dim_emb)
        t_x_dash = torch.sum(torch.stack(lst_x, dim=1), dim=1)

        return t_x_dash


class ProbAdjustedSimpleDecoder(SimpleDecoder):

    def forward(self, prob_c: torch.Tensor):

        # input_c: (N_b, N_digits, N_ary)
        lst_x = []
        n_batch = prob_c.shape[0]
        prob_nonzero_prev = torch.ones((n_batch, 1), dtype=torch.float32, device=prob_c.device, requires_grad=False)
        for d in range(self._n_digits):
            # probs: p(c_d = a, c_d-1 != 0) = p(c_d = a | c_d-1 != 0)*p(c_d-1 != 0)
            probs = prob_c[:, d, :] * prob_nonzero_prev
            # lst_x[n]: (N_b, N_dim_emb)
            lst_x.append(self.lst_c_to_x[d](probs))
            # update nonzero probability
            prob_nonzero_prev = prob_nonzero_prev * torch.sum(prob_c[:, d, 1:], dim=-1, keepdim=True)

        # t_x_dash: \sum_n{(N_b, n, N_dim_emb)} = (N_b, N_dim_emb)
        t_x_dash = torch.sum(torch.stack(lst_x, dim=1), dim=1)

        return t_x_dash
