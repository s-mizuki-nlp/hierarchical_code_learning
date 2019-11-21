#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from entmax.activations import entmax15, sparsemax

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