#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L

### self-supervised loss classes ###

class ReconstructionLoss(L._Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """
        reconstruction loss.
        :return L2(x, x')

        """
        super(ReconstructionLoss, self).__init__(size_average, reduce, reduction)
        # sample-wise & element-wise mean
        self._mse_loss = L.MSELoss(reduction=reduction)

    def forward(self, t_x_dash, t_x):
        return self._mse_loss.forward(t_x_dash, t_x)


### unsupervised loss classes ###

class MutualInformationLoss(L._Loss):

    __EPS = 1E-5

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
        :param probs: probability array (variable shape)
        :return: ent = \sum_{x \in A} p(x)*lnp(x)
        """

        def _p_log_p(_probs):
            return - _probs * torch.log(_probs+self.__EPS)

        ent = _p_log_p(probs) + _p_log_p(1.0 - probs)
        return ent

    def forward(self, t_prob_c_zero: torch.Tensor):

        # t_prob_c_zero: (N_b, N_digits); t_prob_c_zero[b,n] = {p(c_n=0|x_b)}
        # t_prob_c_zero_mean: (N_digits,)
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


### supervised loss classes ###

class CodeLengthPredictionLoss(L._Loss):

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='mean'):

        super(CodeLengthPredictionLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale
        self._mse_loss = L.MSELoss(reduction=reduction)

    def forward(self, t_prob_c_zero: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        evaluates L2 loss of the predicted code length and true code length in a normalized scale.

        :param t_prob_c_zero: probability array of p(c_n=0|x)
        :param y_true: true code length which is normalized to [0,1] range
        """

        # t_prob_c_zero: (N_b, N_digits); t_prob_c_zero[b,n] = p(c_n=0|x_b)
        # y_pred: (N_b,)
        # y_true: (N_b,)
        y_pred = torch.mean(1.0 - t_prob_c_zero, dim=-1)
        loss = self._mse_loss(y_pred, y_true)

        return loss * self._scale


class HyponymyScoreLoss(L._Loss):

    def __init__(self, scale: float = 1.0, normalize_by_digits: bool = True, size_average=None, reduce=None, reduction='mean') -> "HyponymyScoreLoss":

        super(HyponymyScoreLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale
        self._normalize_by_digits = normalize_by_digits
        self._mse_loss = L.MSELoss(reduction=reduction)

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device


    def _intensity_to_probability(self, t_intensity):
        # t_intensity can be either one or two dimensional tensor.
        dtype, device = self._dtype_and_device(t_intensity)
        pad_shape = t_intensity.shape[:-1] + (1,)

        t_pad_begin = torch.zeros(pad_shape, dtype=dtype, device=device)
        t_pad_end = torch.ones(pad_shape, dtype=dtype, device=device)

        t_prob = torch.cumprod(1.0 - torch.cat((t_pad_begin, t_intensity), dim=-1), dim=-1) * torch.cat((t_intensity, t_pad_end), dim=-1)

        return t_prob

    def calc_soft_code_length(self, t_prob_c: torch.Tensor):
        t_p_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0)).squeeze()
        n_digits = t_p_c_zero.shape[-1]
        dtype, device = self._dtype_and_device(t_prob_c)

        t_p_at_n = self._intensity_to_probability(t_p_c_zero)
        t_at_n = torch.arange(n_digits+1, dtype=dtype, device=device)

        ret = torch.sum(t_p_at_n * t_at_n, dim=-1)
        return ret

    def _calc_break_intensity(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # x: hypernym, y: hyponym

        # t_p_c_*_zero: (n_batch, n_digits)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=torch.tensor(0)).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=torch.tensor(0)).squeeze()

        ret = 1.0 - t_p_c_x_zero - torch.sum(t_prob_c_x * t_prob_c_y, dim=-1) + 2 * t_p_c_x_zero * t_p_c_y_zero
        return ret


    def calc_soft_common_prefix_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        t_break_intensity = self._calc_break_intensity(t_prob_c_x, t_prob_c_y)
        t_prob_break = self._intensity_to_probability(t_break_intensity)

        t_at_n = torch.arange(n_digits+1, dtype=dtype, device=device)
        ret = torch.sum(t_prob_break * t_at_n, dim=-1)

        return ret

    def calc_soft_hyponymy_score(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # calculate soft hyponymy score
        # x: hypernym, y: hyponym
        # t_prob_c_*[b,n,v] = Pr{C_n=v|x_b}; t_prob_c_*: (n_batch, n_digits, n_ary)

        # hcl = hypernym code length
        hcl = self.calc_soft_code_length(t_prob_c_x)
        # cpl = common prefix length
        cpl = self.calc_soft_common_prefix_length(t_prob_c_x, t_prob_c_y)

        return cpl - hcl

    def forward(self, t_prob_c_batch: torch.Tensor, lst_hyponym_tuple: List[Tuple[int,int,float]]) -> torch.Tensor:
        """
        evaluates L2 loss of the predicted hyponymy score and true hyponymy score.

        :param t_prob_c_batch: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param lst_hyponym_tuple: list of (hypernym index, hyponym index, hyponymy score) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx_x = torch.LongTensor([tup[0] for tup in lst_hyponym_tuple], device=device)
        t_idx_y = torch.LongTensor([tup[1] for tup in lst_hyponym_tuple], device=device)
        y_true = torch.FloatTensor([tup[2] for tup in lst_hyponym_tuple], device=device)

        t_prob_c_x = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_x)
        t_prob_c_y = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_y)

        y_pred = self.calc_soft_hyponymy_score(t_prob_c_x, t_prob_c_y)

        if self._normalize_by_digits:
            n_digits = t_prob_c_batch.shape[1]
            y_true /= n_digits
            y_pred /= n_digits

        loss = self._mse_loss(y_pred, y_true)

        return loss * self._scale
