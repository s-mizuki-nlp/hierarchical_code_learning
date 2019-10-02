#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from entmax.activations import entmax15

class StraightThroughEstimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(StraightThroughEstimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def forward(self, probs, dim: int = -1):

        if self._add_gumbel_noise:
            logits = torch.log(probs)
            output = F.gumbel_softmax(logits=logits, tau=self._temperature, hard=True, dim=dim)
            return output

        # one-hot operation
        t_z = probs
        index = t_z.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(t_z).scatter_(dim, index, 1.0)
        output = y_hard - t_z.detach() + t_z

        return output


class GumbelSoftmax(nn.Module):

    def __init__(self, temperature: float = 1.0, **kwargs):
        super(GumbelSoftmax, self).__init__()
        self._temperature = temperature

    def forward(self, probs, dim: int = -1):
        logits = torch.log(probs)
        return F.gumbel_softmax(logits=logits, tau=self._temperature, hard=False, dim=dim)


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
        logits = torch.log(probs)
        if self._add_gumbel_noise:
            logits = self._gumbel_noise(logits=logits)

        return entmax15(logits, dim=dim)