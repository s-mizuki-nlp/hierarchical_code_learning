#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Tuple
from contextlib import ExitStack
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from model.autoencoder import AutoEncoder
from model.encoder import SimpleEncoder, CodeLengthAwareEncoder

class UnsupervisedTrainer(object):

    def __init__(self, model: AutoEncoder, loss_reconst: _Loss,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None,
                 ):

        self._optimizer = optimizer
        self._model = model
        self._encoder = model._encoder
        self._decoder = model._decoder
        self._loss_reconst = loss_reconst
        self._loss_mutual_info = loss_mutual_info

    def set_device(self, device):
        self._device = device
        self._model.to(self._device)

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).to(self._device)

    def train_single_step(self, mat_x: np.array, coef_loss_mutual_info: float = 1.0):

        self._optimizer.zero_grad()

        # forward computation
        t_x = self._numpy_to_tensor(mat_x)
        t_intermediate, t_code_prob, t_x_dash = self._model.forward(t_x, requires_grad=True)

        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = 0.0

        loss = loss_reconst + coef_loss_mutual_info * loss_mi
        loss.backward()
        self._optimizer.step()

        dict_losses = {
            "loss_reconst": float(loss_reconst),
            "mutual_info": float(loss_mi),
            "loss_total": float(loss)
        }
        return dict_losses

    def _evaluate_code_representation(self, t_code_prob: torch.Tensor) -> Dict[str, float]:
        return {}

    def evaluate_single_step(self, mat_x: np.array, coef_loss_mutual_info: float = 1.0):

        # forward computation without back-propagation
        t_x = self._numpy_to_tensor(mat_x)
        t_intermediate, t_code_prob, t_x_dash = self._model.forward(t_x, requires_grad=False)

        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)
        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = 0.0

        metrics = {
            "loss_reconst": float(loss_reconst),
            "mutual_info": float(loss_mi),
            "loss_total": float(loss_reconst) + coef_loss_mutual_info * float(loss_mi)
        }
        if t_code_prob is not None:
            metrics_repr = self._evaluate_code_representation(t_code_prob)
            metrics.update(metrics_repr)

        return metrics


class SupervisedCodeLengthTrainer(UnsupervisedTrainer):

    def __init__(self, model: SimpleEncoder, loss_reconst: _Loss,
                 device,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None,
                 loss_code_length: Optional[_Loss] = None):

        super(SupervisedCodeLengthTrainer, self).__init__(model=model, loss_reconst=loss_reconst, device=device,
                                                          optimizer=optimizer, loss_mutual_info=loss_mutual_info)

        self._loss_code_length = loss_code_length


    def train_single_step(self, mat_x: np.array, vec_y: np.array, coef_loss_mutual_info: float = 1.0, coef_loss_supervised: float = 1.0):

        self._optimizer.zero_grad()

        t_x = self._numpy_to_tensor(mat_x)
        t_y = self._numpy_to_tensor(vec_y)

        # forward computation
        t_intermediate, t_code_prob, t_x_dash = self._model.forward(t_x, requires_grad=True)

        loss = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
            loss = loss + coef_loss_mutual_info * loss_mi

        loss_supervised = self._loss_code_length.forward(t_code_prob, t_y)
        loss = loss + coef_loss_supervised * loss_supervised

        loss.backward()
        self._optimizer.step()


class SupervisedHypernymyRelationTrainer(UnsupervisedTrainer):

    def __init__(self, model: SimpleEncoder, loss_reconst: _Loss,
                 device,
                 optimizer: torch.optim.Optimizer,
                 loss_mutual_info: Optional[_Loss] = None,
                 loss_hypernym_relation: Optional[_Loss] = None):

        super(SupervisedHypernymyRelationTrainer, self).__init__(model=model, loss_reconst=loss_reconst, device=device,
                                                                 optimizer=optimizer, loss_mutual_info=loss_mutual_info)

        self._loss_hypernym_relation = loss_hypernym_relation


    def train_single_step(self, mat_x: np.array, mat_y: np.array, coef_loss_mutual_info: float = 1.0, coef_loss_supervised: float = 1.0):

        self._optimizer.zero_grad()

        t_x = self._numpy_to_tensor(mat_x)
        t_y = self._numpy_to_tensor(mat_y)

        # forward computation
        t_intermediate, t_code_prob, t_x_dash = self._model.forward(t_x, requires_grad=True)

        loss = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
            loss = loss + coef_loss_mutual_info * loss_mi

        loss_supervised = self._loss_hypernym_relation.forward(t_code_prob, t_y)
        loss = loss + coef_loss_supervised * loss_supervised

        loss.backward()
        self._optimizer.step()
