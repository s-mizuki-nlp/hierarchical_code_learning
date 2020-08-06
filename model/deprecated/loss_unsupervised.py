#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn.modules import loss as L

### unsupervised loss classes ###
class MutualInformationLoss(L._Loss):

    _EPS = 1E-5

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='mean'):
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
        t_prob_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0, device=t_prob_c.device)).squeeze()
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

    @scale.setter
    def scale(self, value):
        self._scale = value


class OriginalMutualInformationLoss(MutualInformationLoss):

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='mean'):
        """
        mutual information loss. it calculates the negative mutual information multiplied by the specified scale coefficient.
        weight of each digit is calculated based on the `gate_open_ratio` property.
        when `gate_open_ratio` is one (=full_open), uniform weights are applied.
        warning: it always apply a samplewise (=in-batch) mean. you can only choose elementwise mean or sum.
        :return: scale * $-I(C;X)$ = scale * $-(\sum_{n}{w_n(H(C_n) - H(C_n|X))}$

        :param scale:  scale coefficient
        :param size_average:
        :param reduce:
        :param reduction:
        """
        super().__init__(scale, size_average, reduce, reduction)
        self._gate_open_ratio = 1.0

    @property
    def gate_open_ratio(self) -> float:
        return self._gate_open_ratio

    @gate_open_ratio.setter
    def gate_open_ratio(self, value: float):
        self._gate_open_ratio = value

    def _calc_digit_weights(self, n_digits: int, reduction: str, dtype, device) -> torch.Tensor:
        if self._gate_open_ratio == 1.0:
            t_w = torch.ones(size=(n_digits,), dtype=dtype, device=device)
        else:
            t_w = torch.zeros(size=(n_digits,), dtype=dtype, device=device)
            i, d = divmod(self._gate_open_ratio*n_digits, 1)
            i = int(i)
            if i == 0:
                t_w[0] = 1.0
            else:
                t_w[:i] = 1.0; t_w[i] = d

        if reduction.endswith("mean"):
            t_w = t_w / torch.sum(t_w)
        t_w = t_w.reshape(1,-1)
        return t_w

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

        # weight for each digit: (N_digits,)
        # weight is scaled; sum(weight) = 1
        n_digits, dtype, device = t_prob_c.shape[1], t_prob_c.dtype, t_prob_c.device
        weight = self._calc_digit_weights(n_digits=n_digits, reduction=self.reduction, dtype=dtype, device=device)
        mutual_info = torch.sum(weight*(total_entropy - conditional_entropy))

        return - self._scale * mutual_info
