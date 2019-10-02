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

# class HypernymyPairwiseLoss(L._Loss):
#
#     def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='elementwise_mean'):
#
#         super(HypernymyPairwiseLoss, self).__init__(size_average, reduce, reduction)
#
#         self._scale = scale
#
#     def _nonzero_digits_score(self, t_prob_c_hypernym: torch.Tensor, t_prob_c_hyponym: torch.Tensor):
#         # t_prob_c_*: (N_digits, N_ary); t_prob_c_*[n,c] = p(c_n=c)
#         # :return: (N_digits,); return[n] = max(0, p(c_n=0|hyponym) - p(c_n=0|hypernym))
#         return F.relu(t_prob_c_hyponym[:,0] - t_prob_c_hypernym[:,0])
#
#     def _nonzero_digits_overlap_score(self, t_prob_c_hypernym: torch.Tensor, t_prob_c_hyponym: torch.Tensor):
#         # t_prob_c_*: (N_digits, N_ary); t_prob_c_*[n,c] = p(c_n=c)
#         # :return: (N_digits,); return[n] = p(c_n!=0|hypernym) max(0, p(c_n=0|hyponym) - p(c_n=0|hypernym))
#         return F.relu(t_prob_c_hyponym[:,0] - t_prob_c_hypernym[:,0])
#
#     def code_inclusion_score(self, t_prob_c_hypernym: torch.Tensor, t_prob_c_hyponym: torch.Tensor):
#         # t_prob_c_*: (N_digits, N_ary); t_prob_c_*[n,c] = p(c_n=c)
#         t_prob_c_nonzero_hypernym = 1.0 - t_prob_c_hypernym[:,0]
#
#
#
#     def forward(self, t_prob_c: torch.Tensor, lst_hypernymy_tuple: List[Tuple[int, int, float]]):
#
#         # t_prob_c: (N_b, N_digits, N_ary); t_prob_c[b,n,c] = p(c_n=c|x_b)
#         loss = torch.scalar_tensor(0., dtype=torch.float32, requires_grad=True)
#         if len(lst_hypernymy_tuple) == 0:
#             return loss
#
#         for idx_hypernym, idx_hyponym, y_true in lst_hypernymy_tuple:
#             prob_c_hypernym = t_prob_c[idx_hypernym,:,:]
#             prob_c_hyponym = t_prob_c[idx_hyponym,:,:]




### experimental ###

class HypponyLoss(L._Loss):
    pass