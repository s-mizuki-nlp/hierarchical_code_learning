#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from entmax.activations import entmax15, sparsemax
import numpy as np

_EPS = 1E-6

class StraightThroughEstimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(StraightThroughEstimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def forward(self, probs, dim: int = -1):

        if self._add_gumbel_noise:
            logits = torch.log(probs+_EPS)
            output = F.gumbel_softmax(logits=logits, tau=self._temperature, hard=True, dim=dim)
            return output

        # one-hot operation
        t_z = probs
        index = t_z.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(t_z).scatter_(dim, index, 1.0)
        output = y_hard - t_z.detach() + t_z

        return output

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class GumbelSoftmax(nn.Module):

    def __init__(self, temperature: float = 1.0, add_gumbel_noise: bool = True, **kwargs):
        super(GumbelSoftmax, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

        if not add_gumbel_noise:
            Warning("it reduces to a simple softmax activation.")

    def forward(self, probs, dim: int = -1):

        logits = torch.log(probs+_EPS)

        if self._add_gumbel_noise:
            return F.gumbel_softmax(logits=logits, tau=self._temperature, hard=False, dim=dim)
        else:
            return F.softmax(input=logits / self._temperature, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class Entmax15Estimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(Entmax15Estimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def _gumbel_noise(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self._temperature  # ~Gumbel(logits,tau)

        return gumbels

    def forward(self, probs, dim: int = -1):
        logits = torch.log(probs+_EPS)
        if self._add_gumbel_noise:
            logits = self._gumbel_noise(logits=logits)

        return entmax15(logits, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class SparsemaxEstimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(SparsemaxEstimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def _gumbel_noise(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self._temperature  # ~Gumbel(logits,tau)

        return gumbels

    def forward(self, probs, dim: int = -1):
        logits = torch.log(probs+_EPS)
        if self._add_gumbel_noise:
            logits = self._gumbel_noise(logits=logits)

        return sparsemax(logits, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class MaskedGumbelSoftmax(GumbelSoftmax):

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def forward(self, probs, dim: int = -1):

        dtype, device = self._dtype_and_device(probs)
        n_ary = probs.shape[-1]

        # split p(C_n=0|x) and p(C_n!=0|x)
        # t_p_c_zero: (n_batch, n_digits, 1)
        probs_zero = torch.index_select(probs, dim=-1, index=torch.tensor(0, device=device))
        # t_p_c_nonzero: (n_batch, n_digits, n_ary-1)
        probs_nonzero = torch.index_select(probs, dim=-1, index=torch.arange(1, n_ary, dtype=torch.long, device=device))

        # apply gumbel-softmax trick only on nonzero probabilities
        gumbels_nonzero = super().forward(probs_nonzero, dim=-1)

        # concat with zero-probs and nonzero-probs
        y_soft = torch.cat((probs_zero, (1.0-probs_zero)*gumbels_nonzero), dim=-1)

        return y_soft

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class ScheduledGumbelSoftmax(GumbelSoftmax):

    def __init__(self, temperature: float = 1.0, add_gumbel_noise: bool = True, **kwargs):
        super().__init__(temperature, add_gumbel_noise, **kwargs)
        self._gate_open_ratio = 0.0

    @property
    def gate_open_ratio(self) -> float:
        return self._gate_open_ratio

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        self._gate_open_ratio = value

    def _calc_gate_mask(self, n_digits, dtype, device):
        # gate close <=> gate_mask=1, gate open <=> gate_mask=0
        # gate full close <=> gate_open_ratio=0, gate full open <=> gate_open_ratio=1
        # formula: gate_mask[n] = \sigma(\gamma*(n + offset - \alpha * r)
        # it always holds: gate_mask[n+1] > gate_mask[n]
        # when r=0 (=full close), gate_mask[0] = \sigma(\gamma*offset) = 1 - \epsilon
        # when r=1 (=full open), gate_mask[n_digit-1] = \sigma(\gamma*(n_digit-1 + offset - \alpha) = \epsilon
        offset = 0.5
        epsilon = 0.01
        coef_gamma = 2 * np.log((1. - epsilon) / epsilon)

        vec_n = torch.arange(n_digits, dtype=dtype, device=device)
        vec_intercepts = coef_gamma*(vec_n + offset - n_digits * self._gate_open_ratio)
        gate_mask = torch.sigmoid(vec_intercepts)
        gate_mask = torch.clamp(gate_mask, min=epsilon, max=1.0-epsilon)

        return gate_mask

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def forward(self, probs, dim: int = -1):

        dtype, device = self._dtype_and_device(probs)
        if probs.ndimension() == 3:
            n_digits, n_ary = probs.shape[1:]
        else:
            n_digits, n_ary = probs.shape

        # apply gumbel-softmax trick
        probs_gs = super().forward(probs, dim)

        # split p(C_n=0|x) and p(C_n!=0|x)
        # t_p_c_zero: (n_batch, n_digits, 1)
        probs_gs_zero = torch.index_select(probs_gs, dim=-1, index=torch.tensor(0, device=device))
        # t_p_c_nonzero: (n_batch, n_digits, n_ary-1)
        probs_gs_nonzero = torch.index_select(probs_gs, dim=-1, index=torch.arange(1, n_ary, dtype=torch.long, device=device))

        # compute gate mask
        # gate_mask: (1,n_digits,1)
        gate_mask = self._calc_gate_mask(n_digits=n_digits, device=device, dtype=dtype).reshape(1,-1,1)

        # apply gate mask by element-wise max
        probs_gs_zero = torch.max(gate_mask, probs_gs_zero)

        # concat with zero-probs and nonzero-probs
        y_soft = torch.cat((probs_gs_zero, (1.0-probs_gs_zero)*probs_gs_nonzero), dim=-1)

        return y_soft

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
