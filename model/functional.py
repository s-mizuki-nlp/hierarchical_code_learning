#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Function

class CodeProbabiltyAdjuster(Function):
    """
    adjusts code probability p(c_n=m) so that it satisfies monotone-increasing condition of
    assigning zero-value on each digits.
    in short, this function adjusts so that p(c_{n-1}=0) <= p(c_n=0) will be satisifed.
    """

    @staticmethod
    def forward(ctx, t_prob_c: torch.Tensor):

        dtype, device = t_prob_c.dtype, t_prob_c.device
        n_batch, n_digits, n_ary = t_prob_c.shape

        # t_prob_zero_prev: (n_batch,)
        t_prob_zero_prev = torch.zeros((n_batch,), dtype=dtype, device=device)
        lst_t_prob_c_zero_adj = []
        for digit in range(n_digits):
            # t_prob_c_zero_n: (n_batch,)
            t_prob_c_zero_n = t_prob_c[:,digit,0]
            t_prob_c_zero_adj_n = t_prob_zero_prev + t_prob_c_zero_n*(1. - t_prob_zero_prev)
            lst_t_prob_c_zero_adj.append(t_prob_c_zero_adj_n)
            t_prob_zero_prev = t_prob_c_zero_adj_n

        # t_prob_c_zero_adj: (n_batch, n_digits, 1)
        t_prob_c_zero_adj = torch.stack(lst_t_prob_c_zero_adj, dim=1).unsqueeze(-1)

        # t_prob_c_nonzero: (n_batch, n_digits, n_ary-1)
        t_prob_c_nonzero = torch.narrow(t_prob_c, dim=-1, start=1, length=n_ary-1) + 1E-6
        # coef_adj: (n_batch, n_digits, 1)
        coef_adj = (1.-t_prob_c_zero_adj) / torch.sum(t_prob_c_nonzero, dim=-1, keepdim=True)
        # t_prob_c_nonzero_adj: (n_batch, n_digits, n_ary-1)
        t_prob_c_nonzero_adj = coef_adj * t_prob_c_nonzero

        # t_prob_c_adj: (n_batch, n_digits, n_ary)
        t_prob_c_adj = torch.cat((t_prob_c_zero_adj, t_prob_c_nonzero_adj), dim=-1)

        ctx.save_for_backward(t_prob_c_adj)

        return t_prob_c_adj

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result