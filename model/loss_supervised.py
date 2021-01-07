#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import List, Tuple, Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import loss as L


class CodeLengthPredictionLoss(L._Loss):

    def __init__(self, scale: float = 1.0, normalize_code_length: bool = False, normalize_coefficient_for_ground_truth: float = 1.0,
                 distance_metric: str = "scaled-mse",
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
        elif distance_metric == "autoscaled-mse":
            self._func_distance = self._auto_scaled_mse
        elif distance_metric == "positive-autoscaled-mse":
            self._func_distance = self._positive_auto_scaled_mse
        elif distance_metric == "batchnorm-mse":
            self._func_distance = self._batchnorm_mse
            self._m = nn.BatchNorm1d(1)
        elif distance_metric == "mae":
            self._func_distance = self._mae
        elif distance_metric == "scaled-mae":
            self._func_distance = self._scaled_mae
        elif distance_metric == "cosine":
            self._func_distance = self._cosine_distance
        elif distance_metric == "hinge":
            self._func_distance = self._hinge_distance
        elif distance_metric == "binary-cross-entropy":
            self._func_distance = self._bce
        else:
            raise AttributeError(f"unsupported distance metric was specified: {distance_metric}")

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

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

    def _auto_scaled_mse(self, u, v) -> torch.Tensor:
        # assume u and y is predicted and ground-truth values, respectively.
        scale = torch.sum(u.detach()*v.detach()) / (torch.sum(u.detach()**2)+1E-6)
        if scale > 0:
            loss = F.mse_loss(scale*u, v, reduction=self.reduction)
        else:
            loss = self._scaled_mse(u, v)
        return loss

    def _positive_auto_scaled_mse(self, u, v) -> torch.Tensor:
        # assume u and y is predicted and ground-truth values, respectively.
        scale = max(1.0, torch.sum(u.detach()*v.detach()) / (torch.sum(u.detach()**2)+1E-6))
        loss = F.mse_loss(scale*u, v, reduction=self.reduction)
        return loss

    def _mae(self, u, v) -> torch.Tensor:
        return F.l1_loss(u, v, reduction=self.reduction)

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

    def _bce(self, u, v) -> torch.Tensor:
        return F.binary_cross_entropy(u, v, reduction=self.reduction)

    def _intensity_to_probability(self, t_intensity):
        # t_intensity can be either one or two dimensional tensor.
        dtype, device = self._dtype_and_device(t_intensity)
        pad_shape = t_intensity.shape[:-1] + (1,)

        t_pad_begin = torch.zeros(pad_shape, dtype=dtype, device=device)
        t_pad_end = torch.ones(pad_shape, dtype=dtype, device=device)

        t_prob = torch.cumprod(1.0 - torch.cat((t_pad_begin, t_intensity), dim=-1), dim=-1) * torch.cat((t_intensity, t_pad_end), dim=-1)

        return t_prob

    def calc_soft_code_length(self, t_prob_c: torch.Tensor):
        t_p_c_zero = torch.index_select(t_prob_c, dim=-1, index=torch.tensor(0, device=t_prob_c.device)).squeeze()
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

        t_idx = torch.tensor([tup[0] for tup in lst_code_length_tuple], dtype=torch.long, device=device)
        y_true = torch.tensor([tup[1] for tup in lst_code_length_tuple], dtype=dtype, device=device)

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

        t_idx_x = torch.tensor([tup[0] for tup in lst_code_length_diff_tuple], dtype=torch.long, device=device)
        t_idx_y = torch.tensor([tup[1] for tup in lst_code_length_diff_tuple], dtype=torch.long, device=device)
        y_true = torch.tensor([tup[2] for tup in lst_code_length_diff_tuple], dtype=dtype, device=device)

        # compute diff of code length
        t_code_length = self.calc_soft_code_length(t_prob_c=t_prob_c_batch)
        t_code_length_x = torch.index_select(t_code_length, dim=0, index=t_idx_x)
        t_code_length_y = torch.index_select(t_code_length, dim=0, index=t_idx_y)
        # code length diff = len(hyponym:y) - len(hypernym:x)
        y_pred = t_code_length_y - t_code_length_x

        # scale ground-truth value and predicted value
        if self._normalize_code_length:
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
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

        ret = 1.0 - (torch.sum(t_prob_c_x * t_prob_c_y, dim=-1) - t_p_c_x_zero * t_p_c_y_zero)
        return ret

    def calc_ancestor_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()
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

    def calc_soft_lowest_common_ancestor_length(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):
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
        # beta = probability of identity relation
        beta = self.calc_synonym_probability(t_prob_c_x, t_prob_c_y)
        # l_lca = length of the lowest common ancestor
        l_lca = self.calc_soft_lowest_common_ancestor_length(t_prob_c_x, t_prob_c_y)

        score = alpha * (l_hypo - l_hyper) + (1. - (alpha + beta)) * (l_lca - l_hyper)

        return score

    def forward(self, t_prob_c_batch: torch.Tensor, lst_hyponymy_tuple: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param t_prob_c_batch: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param lst_hyponymy_tuple: list of (hypernym index, hyponym index, hyponymy score) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        # clamp values so that it won't produce nan value.
        t_prob_c_batch = torch.clamp(t_prob_c_batch, min=1E-5, max=(1.0-1E-5))

        t_idx_x = torch.tensor([tup[0] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        t_idx_y = torch.tensor([tup[1] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        y_true = torch.tensor([tup[2] for tup in lst_hyponymy_tuple], dtype=dtype, device=device)

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

    def calc_synonym_probability(self, t_prob_c_x: torch.Tensor, t_prob_c_y: torch.Tensor):

        n_digits, n_ary = t_prob_c_x.shape[-2:]
        dtype, device = self._dtype_and_device(t_prob_c_x)

        # t_p_c_*_zero: (n_batch, n_digits)
        idx_zero = torch.tensor(0, device=t_prob_c_x.device)
        t_p_c_x_zero = torch.index_select(t_prob_c_x, dim=-1, index=idx_zero).squeeze()
        t_p_c_y_zero = torch.index_select(t_prob_c_y, dim=-1, index=idx_zero).squeeze()

        # t_gamma_hat: (n_batch, n_digits)
        t_gamma_hat = torch.sum(t_prob_c_x*t_prob_c_y, dim=-1) - t_p_c_x_zero*t_p_c_y_zero
        # prepend 1.0 at the beginning
        # pad_shape: (n_batch, 1)
        pad_shape = t_gamma_hat.shape[:-1] + (1,)
        t_pad_ones = torch.ones(pad_shape, dtype=dtype, device=device)
        # t_gamma: (n_batch, n_digits+1)
        t_gamma = torch.cat((t_pad_ones, t_gamma_hat), dim=-1)

        # t_delta: (n_batch, n_digits)
        t_delta_hat = t_p_c_x_zero*t_p_c_y_zero
        # append 1.0 at the end.
        t_delta = torch.cat((t_delta_hat, t_pad_ones), dim=-1)

        # t_prob: (n_batch)
        t_prob = torch.sum(t_delta*torch.cumprod(t_gamma, dim=-1), dim=-1)

        return t_prob


class LowestCommonAncestorLengthPredictionLoss(HyponymyScoreLoss):

    def forward(self, t_prob_c_batch: torch.Tensor, lst_hyponymy_tuple: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        evaluates loss of the predicted and ground-truth value of the length of the lowest common ancestor.

        :param t_prob_c_batch: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param lst_hyponymy_tuple: list of (hypernym index, hyponym index, length of lowest  common ancestor) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        t_idx_x = torch.tensor([tup[0] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        t_idx_y = torch.tensor([tup[1] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        y_true = torch.tensor([tup[2] for tup in lst_hyponymy_tuple], dtype=dtype, device=device)

        t_prob_c_x = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_x)
        t_prob_c_y = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_y)

        y_pred = self.calc_soft_lowest_common_ancestor_length(t_prob_c_x, t_prob_c_y)

        # scale ground-truth value and predicted value
        if self._normalize_hyponymy_score:
            # scale predicted value by the number of digits. then value range will be (-1, +1)
            n_digits = t_prob_c_batch.shape[1]
            y_pred /= n_digits
            # scale ground-truth value by the user-specified value.
            y_true *= self._normalize_coef_for_gt

        loss = self._func_distance(y_pred, y_true)

        return loss * self._scale


class EntailmentProbabilityLoss(HyponymyScoreLoss):

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='mean',
                 loss_metric: str = "cross_entropy", focal_loss_gamma: float = 1.0, focal_loss_normalize_weight: bool = False) -> None:

        super(EntailmentProbabilityLoss, self).__init__(scale=scale,
                    distance_metric="binary-cross-entropy",
                    size_average=size_average, reduce=reduce, reduction=reduction)
        accepted_loss_metric = ("cross_entropy", "focal_loss", "dice_loss")
        assert loss_metric in accepted_loss_metric, f"`loss_metric` must be one of these: {','.join(accepted_loss_metric)}"
        self._loss_metric = loss_metric
        self._focal_loss_gamma = focal_loss_gamma
        self._focal_loss_normalize_weight = focal_loss_normalize_weight

    def forward(self, t_prob_c_batch: torch.Tensor, lst_hyponymy_tuple: List[Tuple[int, int, float]]) -> torch.Tensor:
        """
        evaluates loss of the predicted hyponymy score and true hyponymy score.

        :param t_prob_c_batch: probability array. shape: (n_batch, n_digits, n_ary), t_prob_c_batch[b,n,m] = p(c_n=m|x_b)
        :param lst_hyponymy_tuple: list of (hypernym index, hyponym index, hyponymy score) tuples
        """

        # x: hypernym, y: hyponym
        dtype, device = self._dtype_and_device(t_prob_c_batch)

        # clamp values so that it won't produce nan value.
        t_prob_c_batch = torch.clamp(t_prob_c_batch, min=1E-5, max=(1.0-1E-5))

        t_idx_x = torch.tensor([tup[0] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        t_idx_y = torch.tensor([tup[1] for tup in lst_hyponymy_tuple], dtype=torch.long, device=device)
        y_hyponymy_score = torch.tensor([tup[2] for tup in lst_hyponymy_tuple], dtype=dtype, device=device)

        t_prob_c_x = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_x)
        t_prob_c_y = torch.index_select(t_prob_c_batch, dim=0, index=t_idx_y)

        y_prob_entail = self.calc_ancestor_probability(t_prob_c_x, t_prob_c_y)
        y_prob_synonym = self.calc_synonym_probability(t_prob_c_x, t_prob_c_y)
        y_prob_other = 1.0 - (y_prob_entail+y_prob_synonym)

        # clamp values so that it won't produce nan value.
        y_prob_entail = torch.clamp(y_prob_entail, min=1E-5, max=(1.0-1E-5))
        y_prob_synonym = torch.clamp(y_prob_synonym, min=1E-5, max=(1.0-1E-5))
        y_prob_other = torch.clamp(y_prob_other, min=1E-5, max=(1.0-1E-5))

        # pick up the probability based on the ground-truth class: {}.
        y_probs = (y_hyponymy_score >= 1.0).float() * y_prob_entail + \
                  (y_hyponymy_score == 0.0).float() * y_prob_synonym + \
                  (y_hyponymy_score <= -1.0).float() * y_prob_other

        if self._loss_metric == "cross_entropy": # cross-entropy loss
            y_weights = torch.ones_like(y_probs, dtype=dtype)
            loss_i = -1.0 * y_weights * torch.log(y_probs)

        elif self._loss_metric == "focal_loss": # focal loss
            y_weights = (1.0 - y_probs)**self._focal_loss_gamma
            if self._focal_loss_normalize_weight:
                y_weights = len(y_probs) * y_weights / torch.sum(y_weights)
            loss_i = -1.0 * y_weights * torch.log(y_probs)

        elif self._loss_metric == "dice_loss": # dice loss [Li+, 2020]
            gamma = 1.0
            adjusted_y_probs = ((1.0 - y_probs)**self._focal_loss_gamma) * y_probs
            loss_i = 1.0 - (2. * adjusted_y_probs + gamma) / (adjusted_y_probs + 1 + gamma)
        else:
            raise NotImplementedError(f"unknown loss metric: {self._loss_metric}")

        # reduction
        if self.reduction == "sum":
            loss = torch.sum(loss_i)
        elif self.reduction.endswith("mean"):
            loss = torch.mean(loss_i)
        elif self.reduction == "none":
            loss = loss_i

        return loss * self._scale
