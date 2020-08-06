#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.nn.modules import loss as L
import numpy as np

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

    @scale.setter
    def scale(self, value):
        self._scale = value

### unsupervised loss classes ###
class CodeLengthMutualInformationLoss(L._Loss):

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
        super(CodeLengthMutualInformationLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale

        # hyponymy score loss: it will be used to calculate cumulative zero probability.
        self._auxiliary = HyponymyScoreLoss()

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        :param probs: array of binary probability (variable shape)
        :return: ent = \sum_{x \in A} p(x)*lnp(x)
        """

        def _p_log_p(_probs):
            return - _probs * torch.log(_probs+self._EPS)

        ent = _p_log_p(probs) + _p_log_p(1.0 - probs)
        return ent

    def calc_code_length_probability(self, t_prob_c_zero: torch.Tensor):
        return self._auxiliary._intensity_to_probability(t_prob_c_zero)

    def forward(self, t_prob_c: torch.Tensor):

        # t_prob_c_zero: (N_b, N_digits); t_prob_c_zero[b,n] = Pr{C_n=0|x_b}
        # t_prob_code_length: (N_b, N_digits+1); t_prob_code_length[b,n] = Pr{len(C)=n|x_b}; n in {0,1,2,...,N_digits}
        # t_prob_code_length_mean: (N_digits,)
        t_prob_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0, device=t_prob_c.device)).squeeze()
        t_prob_code_length = self.calc_code_length_probability(t_prob_c_zero)
        t_prob_code_length_mean = torch.mean(t_prob_code_length, dim=0, keepdim=False)

        # total_entropy: (N_digits,)
        total_entropy = self.entropy(t_prob_code_length_mean)

        # conditional_entropy: (N_digits,)
        conditional_entropy = torch.mean(self.entropy(t_prob_code_length), dim=0)

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


class CodeValueMutualInformationLoss(CodeLengthMutualInformationLoss):

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

    @classmethod
    def _dtype_and_device(cls, t: torch.Tensor):
        return t.dtype, t.device

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

    @classmethod
    def calc_adjusted_code_probability(cls, t_prob_c: torch.Tensor):
        dtype, device = cls._dtype_and_device(t_prob_c)
        t_prob_c_adj = t_prob_c.clone()

        # t_prob_c_zero: (n_batch, n_digits), t_prob_c_zero[b][d] = Pr{C_d=0|x_b}
        t_prob_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0, device=device)).squeeze()
        # t_prob_beta: (n_batch, n_digits), t_prob_beta[b][d] = \prod_{d'=0}^{d}{1-\t_prob_c[b][d'][0]}
        t_prob_beta = torch.cumprod(1.0 - t_prob_c_zero, dim=-1)

        # adjust code probability using t_beta_prob
        # t_prob_c_adj[b][d][0] = 1.0 - \beta[b][d]
        # t_prob_c_adj[b][d][m] = t_prob_c[b][d][m] * \beta[b][d-1]; exclude both first digit(d=0) and zero-value(m=0)
        t_prob_c_adj[:,:,0] = 1.0 - t_prob_beta
        t_prob_c_adj[:,1:,1:] = t_prob_c[:,1:,1:] * t_prob_beta[:,:-1].unsqueeze(dim=-1)

        return t_prob_c_adj

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

        # t_prob_c: (N_b, N_digits, N_ary); t_prob_c[b,n,c] = {p(c_n=c|x_b)} -> {p(C_d=c|C_{0:d-1} != 0)}
        # adjust probability to incorporate the probability of zero of upper digits
        t_prob_c = self.calc_adjusted_code_probability(t_prob_c)

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

    @property
    def gate_open_ratio(self) -> float:
        return self._gate_open_ratio

    @gate_open_ratio.setter
    def gate_open_ratio(self, value: float):
        self._gate_open_ratio = value
