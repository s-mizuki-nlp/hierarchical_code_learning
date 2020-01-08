#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L


class CodeLengthPredictionLoss(L._Loss):

    def __init__(self, scale: float = 1.0, normalize_code_length: bool = True, normalize_coefficient_for_ground_truth: float = 1.0,
                 distance_metric: str = "mse",
                 size_average=None, reduce=None, reduction='mean'):

        super(CodeLengthPredictionLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale
        self._normalize_code_length = normalize_code_length
        self._normalize_coef_for_gt = normalize_coefficient_for_ground_truth

        self._distance_metric = distance_metric
        if distance_metric == "mse":
            self._func_distance = self._mse
        elif distance_metric == "scaled-mse":
            self._func_distance = self._scaled_mse
        elif distance_metric == "standardized-mse":
            self._func_distance = self._standardized_mse
        elif distance_metric == "batchnorm-mse":
            self._func_distance = self._batchnorm_mse
            self._m = nn.BatchNorm1d(1)
        elif distance_metric == "scaled-mae":
            self._func_distance = self._scaled_mae
        elif distance_metric == "cosine":
            self._func_distance = self._cosine_distance
        elif distance_metric == "hinge":
            self._func_distance = self._hinge_distance
        else:
            raise AttributeError(f"unsupported distance metric was specified: {distance_metric}")

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    @property
    def scale(self):
        return self._scale

    def _standardize(self, vec: torch.Tensor, dim=-1):
        means = vec.mean(dim=dim, keepdim=True)
        stds = vec.std(dim=dim, keepdim=True)
        return (vec - means) / (stds + 1E-6)

    def _scale_dynamic(self, vec: torch.Tensor, dim=-1):
        stds = vec.std(dim=dim, keepdim=True)
        return vec / (stds + 1E-6)

    def _mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(u, v, reduction=self.reduction)

    def _scaled_mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(self._scale_dynamic(u), self._scale_dynamic(v), reduction=self.reduction)

    def _standardized_mse(self, u, v) -> torch.Tensor:
        return F.mse_loss(self._standardize(u), self._standardize(v), reduction=self.reduction)

    def _scaled_mae(self, u, v) -> torch.Tensor:
        return F.l1_loss(self._scale_dynamic(u), self._scale_dynamic(v), reduction=self.reduction)

    def _cosine_distance(self, u, v, dim=0, eps=1e-8) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(u, v, dim, eps)

    def _hinge_distance(self, y_pred, y_true) -> torch.Tensor:
        hinge_loss = F.relu(y_pred - y_true)
        if self.reduction == "mean":
            return torch.mean(hinge_loss)
        elif self.reduction == "sum":
            return torch.sum(hinge_loss)
        elif self.reduction == "none":
            return hinge_loss
        else:
            raise NotImplementedError(f"unsupported reduction method was specified: {self.reduction}")

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

    def forward(self, t_prob_c_batch: torch.Tensor, lst_code_length_tuple: List[Tuple[int, float]]) -> torch.Tensor:
        """
        evaluates L2 loss of the predicted code length and true code length in a normalized scale.

        :param t_prob_c_batch: probability array of p(c_n=m|x); (n_batch, n_digits, n_ary)
        :param lst_hyponymy_tuple: list of (entity index, entity depth) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx = torch.LongTensor([tup[0] for tup in lst_code_length_tuple], device=device)
        y_true = torch.FloatTensor([tup[1] for tup in lst_code_length_tuple], device=device)

        # t_prob_c_batch: (N_b, N_digits, N_ary); t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        t_prob_c = torch.index_select(t_prob_c_batch, dim=0, index=t_idx)

        # y_pred: (len(lst_code_length_tuple),)
        # y_true: (len(lst_code_length_tuple),)
        y_pred = self.calc_soft_code_length(t_prob_c=t_prob_c)

        # scale ground-truth value and predicted value
        if self._normalize_code_length:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            n_digits = t_prob_c_batch.shape[1]
            y_pred /= n_digits
            # scale ground-truth value by the user-specified value.
            y_true *= self._normalize_coef_for_gt

        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale


class CodeLengthDiffPredictionLoss(CodeLengthPredictionLoss):

    def forward(self, t_prob_c_batch: torch.Tensor, lst_code_length_diff_tuple: List[Tuple[int, int, float]]) -> torch.Tensor:

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx_x = torch.LongTensor([tup[0] for tup in lst_code_length_diff_tuple], device=device)
        t_idx_y = torch.LongTensor([tup[1] for tup in lst_code_length_diff_tuple], device=device)
        y_true = torch.FloatTensor([tup[2] for tup in lst_code_length_diff_tuple], device=device)

        # compute diff of code length
        t_code_length = self.calc_soft_code_length(t_prob_c=t_prob_c_batch)
        t_code_length_x = torch.index_select(t_code_length, dim=0, index=t_idx_x)
        t_code_length_y = torch.index_select(t_code_length, dim=0, index=t_idx_y)
        # code length diff = len(hyponym:y) - len(hypernym:x)
        y_pred = t_code_length_y - t_code_length_x

        # scale ground-truth value and predicted value
        if self._normalize_hyponymy_score:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            n_digits = t_prob_c_batch.shape[1]
            y_pred /= n_digits
            # scale ground-truth value by the user-specified value.
            y_true *= self._normalize_coef_for_gt

        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale


class HyponymyScoreLoss(CodeLengthPredictionLoss):

    def __init__(self, scale: float = 1.0, normalize_hyponymy_score: bool = False, normalize_coefficient_for_ground_truth: float = 1.0,
                 distance_metric: str = "scaled-mse",
                 size_average=None, reduce=None, reduction='mean') -> None:

        super(HyponymyScoreLoss, self).__init__(scale=scale,
                    normalize_coefficient_for_ground_truth=normalize_coefficient_for_ground_truth,
                    distance_metric=distance_metric,
                    size_average=size_average, reduce=reduce, reduction=reduction)

        self._normalize_hyponymy_score = normalize_hyponymy_score

    def _calc_break_intensity(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        # x: hypernym, y: hyponym

        # t_p_c_*_zero: (n_batch, n_digits)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=torch.tensor(0)).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=torch.tensor(0)).squeeze()

        ret = 1.0 - (torch.sum(t_prob_c_x * t_prob_c_y, dim=-1) - t_p_c_x_zero * t_p_c_y_zero)
        return ret

    def calc_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=torch.tensor(0)).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=torch.tensor(0)).squeeze()
        # t_beta: (n_batch, n_digits)
        t_beta = t_p_c_x_zero*(1.- t_p_c_y_zero)

        # t_gamma_hat: (n_batch, n_digits)
        t_gamma_hat = torch.sum(t_prob_c_x*t_prob_c_y, dim=-1) - t_p_c_x_zero*t_p_c_y_zero
        # prepend 1.0 at the beginning
        # pad_shape: (n_batch, 1)
        pad_shape = t_gamma_hat.shape[:-1] + (1,)
        t_pad_begin = torch.ones(pad_shape, dtype=dtype, device=device)
        # t_gamma: (n_batch, n_digits)
        t_gamma = torch.narrow(torch.cat((t_pad_begin, t_gamma_hat), dim=-1), dim=-1, start=0, length=n_digits)
        # t_prob: (n_batch,)
        t_prob = torch.sum(t_beta*torch.cumprod(t_gamma, dim=-1), dim=-1)

        return t_prob

    def calc_soft_least_common_ancestor_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
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

        # l_hyper, l_hypo = hypernym / hyponym code length
        l_hyper = self.calc_soft_code_length(t_prob_c_x)
        l_hypo = self.calc_soft_code_length(t_prob_c_y)
        # alpha = probability of hyponymy relation
        alpha = self.calc_ancestor_probability(t_prob_c_x, t_prob_c_y)
        # l_lca = length of the least common ancestor
        l_lca = self.calc_soft_least_common_ancestor_length(t_prob_c_x, t_prob_c_y)

        score = alpha * (l_hypo - l_hyper) + (1. - alpha) * (l_lca - l_hyper)

        return score

    def forward(self, t_prob_c_batch: torch.Tensor, lst_hyponymy_tuple: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param t_prob_c_batch: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param lst_hyponymy_tuple: list of (hypernym index, hyponym index, hyponymy score) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx_x = torch.LongTensor([tup[0] for tup in lst_hyponymy_tuple], device=device)
        t_idx_y = torch.LongTensor([tup[1] for tup in lst_hyponymy_tuple], device=device)
        y_true = torch.FloatTensor([tup[2] for tup in lst_hyponymy_tuple], device=device)

        t_prob_c_x = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_x)
        t_prob_c_y = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_y)

        y_pred = self.calc_soft_hyponymy_score(t_prob_c_x, t_prob_c_y)

        # scale ground-truth value and predicted value
        if self._normalize_hyponymy_score:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            n_digits = t_prob_c_batch.shape[1]
            y_pred /= n_digits
            # scale ground-truth value by the user-specified value.
            y_true *= self._normalize_coef_for_gt

        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale