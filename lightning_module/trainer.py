#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Dict, Callable
import warnings
from collections import defaultdict
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
import pytorch_lightning as pl

from sam.sam import SAM

from model.autoencoder import MaskedAutoEncoder
from model.loss_unsupervised import ReconstructionLoss
from model.loss_supervised import HyponymyScoreLoss, CodeLengthPredictionLoss


class UnsupervisedTrainer(pl.LightningModule):

    def __init__(self,
                 model: MaskedAutoEncoder,
                 loss_reconst: _Loss,
                 loss_mutual_info: Optional[_Loss] = None,
                 dataloader_train: Optional[DataLoader] = None,
                 dataloader_val: Optional[DataLoader] = None,
                 dataloader_test: Optional[DataLoader] = None,
                 learning_rate: Optional[float] = 0.001,
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Dict[str, Callable[[float], float]]]] = None,
                 optimizer_class: Optional[Optimizer] = None
                 ):

        super(UnsupervisedTrainer, self).__init__()

        self._scale_loss_reconst = loss_reconst.scale
        self._scale_loss_mi = loss_mutual_info.scale if loss_mutual_info is not None else 1.

        self._model = model
        self._encoder = model._encoder
        self._decoder = model._decoder
        self._loss_reconst = loss_reconst
        self._loss_mutual_info = loss_mutual_info
        self._learning_rate = learning_rate
        self._dataloaders = {
            "train": dataloader_train,
            "val": dataloader_val,
            "test": dataloader_test
        }
        self._optimizer_class = optimizer_class
        # auxiliary function that is solely used for validation
        self._auxiliary = HyponymyScoreLoss()

        # set model parameter scheduler
        if model_parameter_schedulers is None:
            self._model_parameter_schedulers = {}
        else:
            self._model_parameter_schedulers = model_parameter_schedulers

        if loss_parameter_schedulers is None:
            self._loss_parameter_schedulers = {}
        else:
            self._loss_parameter_schedulers = loss_parameter_schedulers

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).to(self._device)

    def _get_model_device(self):
        return (next(self._model.parameters())).device

    def configure_optimizers(self):
        if self._optimizer_class is None:
            opt = Adam(self.parameters(), lr=self._learning_rate)
        elif self._optimizer_class.__name__ == "SAM":
            opt = self._optimizer_class(self.parameters(), Adam, lr=self._learning_rate)
        else:
            opt = self._optimizer_class(params=self.parameters(), lr=self._learning_rate)
        return opt

    def _asssign_null_collate_function(self, dataloader: Optional[DataLoader]):
        if dataloader is None:
            return dataloader
        if (dataloader.collate_fn is None) or (dataloader.collate_fn is default_collate):
            warnings.warn("update collate_fn with null function.")
            dataloader.collate_fn = lambda v:v
        return dataloader

    @pl.data_loader
    def tng_dataloader(self):
        self._dataloaders["train"] = self._asssign_null_collate_function(self._dataloaders["train"])
        return self._dataloaders["train"]

    @pl.data_loader
    def val_dataloader(self):
        self._dataloaders["val"] = self._asssign_null_collate_function(self._dataloaders["val"])
        return self._dataloaders["val"]

    @pl.data_loader
    def test_dataloader(self):
        self._dataloaders["test"] = self._asssign_null_collate_function(self._dataloaders["test"])
        return self._dataloaders["test"]

    def forward(self, x):
        return self._model.forward(x)

    def training_step(self, data_batch, batch_nb):

        current_step = self.trainer.global_step / (self.trainer.max_nb_epochs * self.trainer.total_batches)
        self._update_model_parameters(current_step, verbose=False)
        self._update_loss_parameters(current_step, verbose=False)

        if isinstance(data_batch, list):
            data_batch = data_batch[0]

        # forward computation
        t_x = torch.tensor(data_batch["embedding"], dtype=torch.float32, device=self._get_model_device()).squeeze(dim=0)
        t_latent_code, t_code_prob, t_x_dash = self._model.forward(t_x)

        # (required) reconstruction loss
        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)

        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        loss = loss_reconst + loss_mi

        dict_losses = {
            "train_loss_reconst": loss_reconst / self._scale_loss_reconst,
            "train_loss_mutual_info": loss_mi / self._scale_loss_mi,
            "train_loss": loss
        }
        return {"loss":loss, "log": dict_losses}

    def _evaluate_code_stats(self, t_code_prob):

        _EPS = 1E-6
        n_ary = self._model.n_ary
        soft_code_length = self._auxiliary.calc_soft_code_length(t_code_prob)
        code_probability_divergence = torch.mean(np.log(n_ary) + torch.sum(t_code_prob * torch.log(t_code_prob + _EPS), axis=-1), axis=-1)

        metrics = {
            "val_soft_code_length_mean":torch.mean(soft_code_length),
            "val_soft_code_length_std":torch.std(soft_code_length),
            "val_code_probability_divergence":torch.mean(code_probability_divergence)
        }
        return metrics

    def validation_step(self, data_batch, batch_nb):

        # forward computation without back-propagation
        t_x = torch.tensor(data_batch[0]["embedding"], dtype=torch.float32, device=self._get_model_device()).squeeze(dim=0)
        t_intermediate, t_code_prob, t_x_dash = self._model._predict(t_x)

        loss_reconst = self._loss_reconst.forward(t_x_dash, t_x)
        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        loss = loss_reconst + loss_mi

        metrics = {
            "val_loss_reconst": loss_reconst / self._scale_loss_reconst,
            "val_mutual_info": loss_mi / self._scale_loss_mi,
            "val_loss": loss
        }
        # if self._loss_mutual_info is not None:
        metrics_repr = self._evaluate_code_stats(t_code_prob)
        metrics.update(metrics_repr)

        return {"val_loss":loss, "log":metrics}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_metrics = defaultdict(list)
        for output in outputs:
            for key, value in output["log"].items():
                avg_metrics[key].append(value)
        for key, values in avg_metrics.items():
            avg_metrics[key] = torch.stack(values).mean()
        return {'avg_val_loss': avg_loss, 'log': avg_metrics}

    def on_save_checkpoint(self, checkpoint):
        device = self._get_model_device()
        if device != torch.device("cpu"):
            # convert device to cpu. it changes self._model instance itself.
            _ = self._model.to(device=torch.device("cpu"))
        # save model dump
        checkpoint["model_dump"] = pickle.dumps(self._model)
        # then revert back if necessary.
        if device != torch.device("cpu"):
            # revert to original device (probably cuda).
            _ = self._model.to(device=device)

    @classmethod
    def load_model_from_checkpoint(cls, weights_path: str, on_gpu: bool, map_location=None, fix_model_missing_attributes: bool = True):
        if on_gpu:
            if map_location is not None:
                checkpoint = torch.load(weights_path, map_location=map_location)
            else:
                checkpoint = torch.load(weights_path)
        else:
            checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        model = pickle.loads(checkpoint["model_dump"])
        if on_gpu:
            model = model.cuda(device=map_location)
        state_dict = {key.replace("_model.", ""):param for key, param in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)

        # fix model attributes for backward compatibility.
        if fix_model_missing_attributes:
            model = cls.fix_missing_attributes(model)

        return model

    @classmethod
    def fix_missing_attributes(cls, model):
        # fix for #9f6ec90, #fbf51f8, #d882d6f, #cd01f68, #d4f59fa
        if getattr(model._encoder, "internal_layer_class_type", None) is None:
            if model._encoder.__class__.__name__ == "AutoRegressiveLSTMEncoder":
                if not hasattr(model._encoder, "_input_transformation"):
                    setattr(model._encoder, "_input_transformation", "time_distributed")
                if not hasattr(model._encoder, "_prob_zero_monotone_increasing"):
                    setattr(model._encoder, "_prob_zero_monotone_increasing", False)
                if not hasattr(model._encoder, "_output_embedding"):
                    setattr(model._encoder, "_output_embedding", "time_distributed")
            elif model._encoder.__class__.__name__ == "TransformerEncoder":
                if not hasattr(model._encoder, "_prob_zero_monotone_increasing"):
                    setattr(model._encoder, "_prob_zero_monotone_increasing", False)
            else:
                model._encoder._internal_layer_class_type = model._encoder.lst_h_to_z[0].__class__.__name__

        return model

    def _update_model_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_nb_epochs

        for parameter_name, scheduler_function in self._model_parameter_schedulers.items():
            if scheduler_function is None:
                continue

            current_value = getattr(self._model, parameter_name, None)
            if current_value is not None:
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(self._model, parameter_name, new_value)

                if verbose:
                    print(f"{parameter_name}: {current_value:.2f} -> {new_value:.2f}")

    def _update_loss_parameters(self, current_step: Optional[float] = None, verbose: bool = False):
        if current_step is None:
            current_step = self.current_epoch / self.trainer.max_nb_epochs

        for loss_name, dict_property_scheduler in self._loss_parameter_schedulers.items():
            # get loss layer
            if not loss_name.startswith("_"):
                loss_name = "_" + loss_name
            loss_layer = getattr(self, loss_name, None)
            if loss_layer is None:
                continue

            # get property name and apply scheduler function
            for property_name, scheduler_function in dict_property_scheduler.items():
                if scheduler_function is None:
                    continue

                # check if property exists
                if not hasattr(loss_layer, property_name):
                    continue

                current_value = getattr(loss_layer, property_name, None)
                new_value = scheduler_function(current_step, self.current_epoch)
                setattr(loss_layer, property_name, new_value)

                if verbose:
                    print(f"{loss_name}.{property_name}: {current_value:.2f} -> {new_value:.2f}")


    def on_epoch_start(self):
        pass


class SupervisedTrainer(UnsupervisedTrainer):

    def __init__(self,
                 model: MaskedAutoEncoder,
                 loss_reconst: ReconstructionLoss,
                 loss_hyponymy: HyponymyScoreLoss,
                 loss_mutual_info: Optional[_Loss] = None,
                 loss_non_hyponymy: Optional[HyponymyScoreLoss] = None,
                 loss_code_length: Optional[CodeLengthPredictionLoss] = None,
                 dataloader_train: Optional[DataLoader] = None,
                 dataloader_val: Optional[DataLoader] = None,
                 dataloader_test: Optional[DataLoader] = None,
                 learning_rate: Optional[float] = 0.001,
                 optimizer_class: Optional[Optimizer] = None,
                 use_intermediate_representation: bool = False,
                 model_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 loss_parameter_schedulers: Optional[Dict[str, Callable[[float], float]]] = None,
                 shuffle_hyponymy_dataset_on_every_epoch: bool = True
                 ):

        super().__init__(model, loss_reconst, loss_mutual_info, dataloader_train, dataloader_val, dataloader_test, learning_rate,
                         model_parameter_schedulers, loss_parameter_schedulers, optimizer_class)

        self._use_intermediate_representation = use_intermediate_representation
        self._loss_hyponymy = loss_hyponymy
        self._loss_non_hyponymy = loss_non_hyponymy
        self._loss_code_length = loss_code_length

        self._scale_loss_hyponymy = loss_hyponymy.scale
        self._scale_loss_non_hyponymy = loss_non_hyponymy.scale if loss_non_hyponymy is not None else loss_hyponymy.scale
        self._scale_loss_code_length = loss_code_length.scale if loss_code_length is not None else 1.

        self._shuffle_hyponymy_dataset_on_every_epoch = shuffle_hyponymy_dataset_on_every_epoch

    def training_step(self, data_batch, batch_nb):

        current_step = self.trainer.global_step / (self.trainer.max_nb_epochs * self.trainer.total_batches)
        self._update_model_parameters(current_step, verbose=False)
        self._update_loss_parameters(current_step, verbose=False)

        if isinstance(data_batch, list):
            data_batch = data_batch[0]

        # forward computation
        t_x = torch.tensor(data_batch["embedding"], dtype=torch.float32, device=self._get_model_device()).squeeze(dim=0)

        # DEBUG
        # return {"loss":torch.tensor(0.0, requires_grad=True), "log":{}}

        t_latent_code, t_code_prob, t_x_dash = self._model.forward(t_x)

        # (required) reconstruction loss
        loss_reconst = self._loss_reconst(t_x_dash, t_x)

        # hyponymy relation related loss
        if self._use_intermediate_representation:
            code_repr = t_latent_code
        else:
            code_repr = t_code_prob

        # (required) hyponymy score loss
        lst_tup_hyponymy = data_batch["hyponymy_relation"]
        loss_hyponymy = self._loss_hyponymy(code_repr, lst_tup_hyponymy)

        # (optional) non-hyponymy score loss
        if self._loss_non_hyponymy is not None:
            lst_tup_non_hyponymy = data_batch["non_hyponymy_relation"]
            loss_non_hyponymy = self._loss_non_hyponymy(code_repr, lst_tup_non_hyponymy)
        else:
            loss_non_hyponymy = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        # (optional) code length loss
        if self._loss_code_length is not None:
            lst_tup_entity_depth = data_batch["entity_depth"]
            loss_code_length = self._loss_code_length(code_repr, lst_tup_entity_depth)
        else:
            loss_code_length = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        # (optional) mutual information loss
        if self._loss_mutual_info is not None:
            loss_mutual_info = self._loss_mutual_info(t_code_prob)
        else:
            loss_mutual_info = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        loss = loss_reconst + loss_hyponymy + loss_non_hyponymy + loss_code_length + loss_mutual_info

        dict_losses = {
            "train_loss_reconst": loss_reconst / self._scale_loss_reconst,
            "train_loss_mutual_info": loss_mutual_info / self._scale_loss_mi,
            "train_loss_hyponymy": loss_hyponymy / self._scale_loss_hyponymy,
            "train_loss_non_hyponymy": loss_non_hyponymy / self._scale_loss_non_hyponymy,
            "train_loss_code_length": loss_code_length / self._scale_loss_code_length,
            "train_loss": loss
        }
        return {"loss":loss, "log": dict_losses}

    def validation_step(self, data_batch, batch_nb):

        if isinstance(data_batch, list):
            data_batch = data_batch[0]

        # forward computation without back-propagation
        t_x = torch.tensor(data_batch["embedding"], dtype=torch.float32, device=self._get_model_device()).squeeze(dim=0)
        t_latent_code, t_code_prob, t_x_dash = self._model._predict(t_x)

        # (required) reconstruction loss
        loss_reconst = self._loss_reconst(t_x_dash, t_x)

        # hyponymy relation related loss
        if self._use_intermediate_representation:
            code_repr = t_latent_code
        else:
            code_repr = t_code_prob

        # (required) hyponymy score loss
        lst_tup_hyponymy = data_batch["hyponymy_relation"]
        loss_hyponymy = self._loss_hyponymy(code_repr, lst_tup_hyponymy)

        # (optional) non-hyponymy score loss
        if self._loss_non_hyponymy is not None:
            lst_tup_non_hyponymy = data_batch["non_hyponymy_relation"]
            loss_non_hyponymy = self._loss_non_hyponymy(code_repr, lst_tup_non_hyponymy)
            loss_hyponymy_positive = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)
        else:
            cache = self._loss_hyponymy.reduction
            self._loss_hyponymy.reduction = "none"
            lst_loss_hyponymy = self._loss_hyponymy(code_repr, lst_tup_hyponymy)
            self._loss_hyponymy.reduction = cache
            n_sample = len(lst_tup_hyponymy)
            loss_non_hyponymy = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device); n_non_hyponymy = 0
            loss_hyponymy_positive = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device); n_hyponymy_positive = 0
            for l, (u, v, distance) in zip(lst_loss_hyponymy, lst_tup_hyponymy):
                if distance <= 0:
                    loss_non_hyponymy += l
                    n_non_hyponymy += 1
                else:
                    loss_hyponymy_positive += l
                    n_hyponymy_positive += 1
            loss_non_hyponymy = loss_non_hyponymy / n_non_hyponymy
            loss_hyponymy_positive = loss_hyponymy_positive / n_hyponymy_positive

        # (optional) code length loss
        if self._loss_code_length is not None:
            lst_tup_entity_depth = data_batch["entity_depth"]
            loss_code_length = self._loss_code_length(code_repr, lst_tup_entity_depth)
        else:
            loss_code_length = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        # (optional) mutual information loss
        if self._loss_mutual_info is not None:
            loss_mi = self._loss_mutual_info(t_code_prob)
        else:
            loss_mi = torch.tensor(0.0, dtype=torch.float32, device=t_code_prob.device)

        loss = loss_reconst + loss_hyponymy + loss_non_hyponymy + loss_code_length + loss_mi

        metrics = {
            "val_loss_reconst": loss_reconst / self._scale_loss_reconst,
            "val_loss_mutual_info": loss_mi / self._scale_loss_mi,
            "val_loss_hyponymy": loss_hyponymy / self._scale_loss_hyponymy,
            "val_loss_non_hyponymy": loss_non_hyponymy / self._scale_loss_non_hyponymy,
            "val_loss_hyponymy_positive": loss_hyponymy_positive / self._scale_loss_hyponymy, # self._scale_loss_non_hyponymy,
            "val_loss_code_length": loss_code_length / self._scale_loss_code_length,
            "val_loss": loss
        }
        metrics_repr = self._evaluate_code_stats(t_code_prob)
        metrics.update(metrics_repr)

        return {"val_loss":loss, "log":metrics}

    def on_epoch_start(self):
        if self._shuffle_hyponymy_dataset_on_every_epoch:
            self.train_dataloader().dataset.shuffle_hyponymy_dataset()


class SupervisedTrainerSAM(SupervisedTrainer):

    def on_sanity_check_start(self):
        self.configure_optimizers()

    def training_step(self, data_batch, batch_nb):
        optimizer = self.optimizers[0]
        assert isinstance(optimizer, SAM), "optimizer must be SAM optimizer class."

        random_seed = np.random.randint(np.iinfo(np.int32).max)

        # first forward-backward pass
        torch.manual_seed(random_seed)
        dict_loss = super().training_step(data_batch, batch_nb)
        dict_loss["loss"].backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        torch.manual_seed(random_seed)
        dict_loss = super().training_step(data_batch, batch_nb)
        dict_loss["loss"].backward()
        optimizer.second_step(zero_grad=True)

        return dict_loss

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # do nothing
        pass

    def backward(self, use_amp, loss, optimizer):
        pass
