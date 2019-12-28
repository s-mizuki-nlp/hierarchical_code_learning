#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn.modules import loss as L

### self-supervised loss classes ###
from model.loss_supervised import HyponymyScoreLoss


class ReconstructionLoss(L._Loss):

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='mean'):
        """
        reconstruction loss.
        :return L2(x, x')

        """
        super(ReconstructionLoss, self).__init__(size_average, reduce, reduction)
        # sample-wise & element-wise mean
        self._mse_loss = L.MSELoss(reduction=reduction)
        self._scale = scale

    def forward(self, t_x_dash, t_x):
        return self._mse_loss.forward(t_x_dash, t_x) * self._scale

    @property
    def scale(self):
        return self._scale

### unsupervised loss classes ###
class MutualInformationLoss(L._Loss):

    _EPS = 1E-5

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='elementwise_mean'):
        """
        mutual information loss. it calculates the negative mutual information multiplied by the specified scale coefficient.
        warning: it always apply a samplewise (=in-batch) mean. you can only choose elementwise mean or sum.
        :return: scale * $-I(C;X)$ = scale * $-(\mean{H(1{C_n=0}) - H(1{C_n=0|X}}$

        :param scale:  scale coefficient
        :param size_average:
        :param reduce:
        :param reduction:
        """
        super(MutualInformationLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        :param probs: array of binary probability (variable shape)
        :return: ent = \sum_{x \in A} p(x)*lnp(x)
        """

        def _p_log_p(_probs):
            return - _probs * torch.log(_probs+self._EPS)

        ent = _p_log_p(probs) + _p_log_p(1.0 - probs)
        return ent

    def forward(self, t_prob_c: torch.Tensor):

        # t_prob_c_zero: (N_b, N_digits); t_prob_c_zero[b,n] = {p(c_n=0|x_b)}
        # t_prob_c_zero_mean: (N_digits,)
        t_prob_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0)).squeeze()
        t_prob_c_zero_mean = torch.mean(t_prob_c_zero, dim=0, keepdim=False)

        # total_entropy: (N_digits,)
        total_entropy = self.entropy(t_prob_c_zero_mean)

        # conditional_entropy: (N_digits,)
        conditional_entropy = torch.mean(self.entropy(t_prob_c_zero), dim=0)

        if self.reduction in ("elementwise_mean", "mean"):
            mutual_info = torch.mean(total_entropy - conditional_entropy)
        elif self.reduction == "sum":
            mutual_info = torch.sum(total_entropy - conditional_entropy)

        return - self._scale * mutual_info

    @property
    def scale(self):
        return self._scale


class OriginalMutualInformationLoss(MutualInformationLoss):

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='elementwise_mean'):
        """
        mutual information loss. it calculates the negative mutual information multiplied by the specified scale coefficient.
        warning: it always apply a samplewise (=in-batch) mean. you can only choose elementwise mean or sum.
        :return: scale * $-I(C;X)$ = scale * $-(\mean{H(C_n) - H(C_n|X)}$

        :param scale:  scale coefficient
        :param size_average:
        :param reduce:
        :param reduction:
        """
        super().__init__(scale, size_average, reduce, reduction)

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        :param probs: array of multi-class probability with variable shape
        :return: ent = \sum_{x_n \in A} p(x_n)*lnp(x_n) n_dim = probs.n_dim - 1
        """

        def _p_log_p(_probs):
            return - _probs * torch.log(_probs+self._EPS)

        ent = torch.sum(_p_log_p(probs), dim=-1)
        return ent

    def forward(self, t_prob_c: torch.Tensor):

        # t_prob_c: (N_b, N_digits, N_ary); t_prob_c[b,n,c] = {p(c_n=c|x_b)}
        # t_prob_c_zero_mean: (N_digits, N_ary)
        t_prob_c_mean = torch.mean(t_prob_c, dim=0, keepdim=False)

        # total_entropy: (N_digits,)
        total_entropy = self.entropy(t_prob_c_mean)

        # conditional_entropy: (N_digits,)
        conditional_entropy = torch.mean(self.entropy(t_prob_c), dim=0)

        if self.reduction in ("elementwise_mean", "mean"):
            mutual_info = torch.mean(total_entropy - conditional_entropy)
        elif self.reduction == "sum":
            mutual_info = torch.sum(total_entropy - conditional_entropy)

        return - self._scale * mutual_info