#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
import argparse

from typing import Optional, Dict, Callable, Iterable, Any, List
import pydash
import numpy as np
from torch.utils.data import DataLoader
from dataset.word_embeddings import AbstractWordEmbeddingsDataset
from model.autoencoder import AutoEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

class CodeLengthEvaluator(object):

    _default_evaluator = {
        "accuracy": lambda y_true, y_pred, **kwargs: accuracy_score(y_true, y_pred),
        "confusion_matrix": lambda y_true, y_pred, **kwargs: confusion_matrix(y_true, y_pred, **kwargs),
        "classification_report": lambda y_true, y_pred, **kwargs: classification_report(y_true, y_pred, **kwargs),
        "macro_f_value": lambda y_true, y_pred, **kwargs: f1_score(y_true, y_pred, average="macro", **kwargs),
    }

    def __init__(self, model: AutoEncoder, dataset: Optional[AbstractWordEmbeddingsDataset] = None, data_loader: Optional[DataLoader] = None,
                 **kwargs_dataloader):

        self._model = model
        if dataset is not None:
            self._dataset = dataset
            self._data_loader = DataLoader(dataset, **kwargs_dataloader)
        elif data_loader is not None:
            self._dataset = data_loader.dataset
            self._data_loader = data_loader

        self._default_evaluator["scaled_mean_absolute_error"] = self._scaled_mean_absolute_error

    def _scaled_mean_absolute_error(self, y_true, y_pred, **kwargs):
        return np.mean(np.abs(y_pred/self._model.n_digits - y_true/np.max(y_true)))

    def evaluate(self, embedding_key_name: str = "embedding",
                    ground_truth_key_path: str = "entity_info.code_length",
                    evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                    **kwargs_for_metric_function):

        evaluator = self._default_evaluator if evaluator is None else evaluator

        lst_code_length = []
        lst_code_length_gt = []
        for batch in self._data_loader:
            t_x = pydash.objects.get(batch, embedding_key_name)
            t_code_length_gt = pydash.objects.get(batch, ground_truth_key_path)

            t_code = self._model._encode(t_x)

            v_code_length = np.count_nonzero(t_code, axis=-1)
            v_code_length_gt = t_code_length_gt.data.numpy()

            lst_code_length.append(v_code_length)
            lst_code_length_gt.append(v_code_length_gt)

        v_code_length = np.concatenate(lst_code_length)
        v_code_length_gt = np.concatenate(lst_code_length_gt)

        dict_ret = {}

        for metric_name, f_metric in evaluator.items():
            dict_ret[metric_name] = f_metric(v_code_length_gt, v_code_length, **kwargs_for_metric_function)

        return v_code_length_gt, v_code_length, dict_ret