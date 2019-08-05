#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from model.encoder import SimpleEncoder, CodeLengthAwareEncoder

class UnsupervisedTrainer(object):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_reconst: _Loss,
                 device,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None):

        self._device = device
        self._optimizer = optimizer
        self._encoder = encoder
        self._decoder = decoder
        self._loss_reconst = loss_reconst
        self._loss_mutual_info = loss_mutual_info

    def _forward(self, mat_x: np.array):

        t_x = torch.from_numpy(mat_x).to(self._device)

        # encoder
        if isinstance(self._encoder, SimpleEncoder):
            t_intermediate = self._encoder.forward(t_x)
            t_code_prob = None
            # decoder
            t_x_dash = self._decoder.forward(t_intermediate)
        else:
            raise NotImplementedError(f"not implemented yet: {self._encoder.__class__.__name__}")

        return t_x, t_intermediate, t_code_prob, t_x_dash


    def train_single_step(self, mat_x: np.array, coef_loss_mutual_info: float = 1.0):

        self._optimizer.zero_grad()

        t_x, t_intermediate, t_code_prob, t_x_dash = self._forward(mat_x)

        loss = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
            loss = loss + coef_loss_mutual_info * loss_mi

        loss.backward()
        self._optimizer.step()


class SupervisedCodeLengthTrainer(UnsupervisedTrainer):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_reconst: _Loss,
                 device,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None,
                 loss_code_length: Optional[_Loss] = None):

        super(SupervisedCodeLengthTrainer, self).__init__(encoder, decoder, loss_reconst, device, optimizer, loss_mutual_info)

        self._loss_code_length = loss_code_length


    def train_single_step(self, mat_x: np.array, vec_y: np.array, coef_loss_mutual_info: float = 1.0, coef_loss_supervised: float = 1.0):

        self._optimizer.zero_grad()

        t_x, t_intermediate, t_code_prob, t_x_dash = self._forward(mat_x)

        loss = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
            loss = loss + coef_loss_mutual_info * loss_mi

        t_y = torch.from_numpy(vec_y).to(self._device)
        loss_supervised = self._loss_code_length.forward(t_code_prob, t_y)
        loss = loss + coef_loss_supervised * loss_supervised

        loss.backward()
        self._optimizer.step()


class SupervisedHypernymyRelationTrainer(UnsupervisedTrainer):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, loss_reconst: _Loss,
                 device,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None,
                 loss_hypernym_relation: Optional[_Loss] = None):

        super(SupervisedHypernymyRelationTrainer, self).__init__(encoder, decoder, loss_reconst, device, optimizer, loss_mutual_info)

        self._loss_hypernym_relation = loss_hypernym_relation


    def train_single_step(self, mat_x: np.array, mat_y: np.array, coef_loss_mutual_info: float = 1.0, coef_loss_supervised: float = 1.0):

        self._optimizer.zero_grad()

        t_x, t_intermediate, t_code_prob, t_x_dash = self._forward(mat_x)

        loss = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
            loss = loss + coef_loss_mutual_info * loss_mi

        t_y = torch.from_numpy(mat_y).to(self._device)
        loss_supervised = self._loss_hypernym_relation.forward(t_code_prob, t_y)
        loss = loss + coef_loss_supervised * loss_supervised

        loss.backward()
        self._optimizer.step()
