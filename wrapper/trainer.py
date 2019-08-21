#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Tuple
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
import pytorch_lightning as pl

from model.autoencoder import AutoEncoder
from model.encoder import SimpleEncoder, CodeLengthAwareEncoder

class UnsupervisedTrainer(pl.LightningModule):

    def __init__(self,
                 model: AutoEncoder, loss_reconst: _Loss,
                 loss_mutual_info: Optional[_Loss] = None,
                 coef_loss_mutual_info: Optional[float] = 1.0,
                 dataloader_train: Optional[DataLoader] = None,
                 dataloader_val: Optional[DataLoader] = None,
                 dataloader_test: Optional[DataLoader] = None,
                 learning_rate: Optional[float] = 0.001
                 ):

        super(UnsupervisedTrainer, self).__init__()

        self._model = model
        self._encoder = model._encoder
        self._decoder = model._decoder
        self._loss_reconst = loss_reconst
        self._loss_mutual_info = loss_mutual_info
        self._coef_loss_mutual_info = coef_loss_mutual_info
        self._learning_rate = learning_rate
        self._dataloaders = {
            "train": dataloader_train,
            "val": dataloader_val,
            "test": dataloader_test
        }

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).to(self._device)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self._learning_rate)
        return opt

    @pl.data_loader
    def tng_dataloader(self):
        return self._dataloaders["train"]

    @pl.data_loader
    def val_dataloader(self):
        return self._dataloaders["val"]

    @pl.data_loader
    def test_dataloader(self):
        return self._dataloaders["test"]

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, data_batch, batch_nb):

        # forward computation
        t_x = data_batch["embedding"]
        t_intermediate, t_code_prob, t_x_dash = self._model.forward(t_x)

        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32)

        loss = loss_reconst + self._coef_loss_mutual_info * loss_mi

        dict_losses = {
            "loss_reconst": float(loss_reconst),
            "mutual_info": float(loss_mi),
            "loss": loss
        }
        return dict_losses

    def _evaluate_code_stats(self, t_code_prob):

        n_ary = self._model.n_ary
        effective_code_length = torch.sum(1.0 - t_code_prob[:,:,0], axis=-1)
        code_divergence = torch.mean(np.log(n_ary) + torch.sum(t_code_prob * torch.log(t_code_prob), axis=-1), axis=-1)

        metrics = {
            "val_effective_code_length_mean":torch.mean(effective_code_length),
            "val_effective_code_length_std":torch.std(effective_code_length),
            "val_code_divergence":torch.mean(code_divergence)
        }
        return metrics

    def validation_step(self, data_batch, batch_nb):

        # forward computation without back-propagation
        t_x = data_batch["embedding"]
        t_intermediate, t_code_prob, t_x_dash = self._model._predict(t_x)

        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)
        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32)

        loss = loss_reconst + self._coef_loss_mutual_info * loss_mi

        metrics = {
            "val_loss_reconst": loss_reconst,
            "val_mutual_info": loss_mi,
            "val_loss": loss
        }
        if self._loss_mutual_info is not None:
            metrics_repr = self._evaluate_code_stats(t_code_prob)
            metrics.update(metrics_repr)

        return metrics

    def validation_end(self, outputs):

        tqdm_dic = defaultdict(float)
        for output in outputs:
            for variable, value in output.items():
                tqdm_dic[variable] += value.item()
        n_output = len(outputs)
        for variable in output.keys():
            tqdm_dic[variable] /= n_output

        return tqdm_dic

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_dump"] = pickle.dumps(self._model)

    @classmethod
    def load_model_from_checkpoint(self, weights_path: str, on_gpu, map_location=None):
        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        model = pickle.loads(checkpoint["model_dump"])
        state_dict = {key.replace("_model.", ""):param for key, param in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)

        return model


class SupervisedCodeLengthTrainer(UnsupervisedTrainer):
    pass

class SupervisedHypernymyRelationTrainer(UnsupervisedTrainer):
    pass