#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import DataLoader
from model.autoencoder import AutoEncoder
from .utils import total_variation_distance


class TotalVariationDistanceEvaluator(object):

    def __init__(self, model: AutoEncoder, data_loader: DataLoader):

        self._model = model
        self._data_loader = data_loader

    def total_variation_distance(self, embedding_key_name="embedding", average: bool = True):

        n_eval = len(self._data_loader.dataset)
        n_ary = self._model.n_ary
        tvd_sum = 0
        for batch in self._data_loader:
            t_emb = batch[embedding_key_name]

            # encode with continuous and discrete
            t_code_discrete = self._model._encode(t_emb)
            _, t_code_continuous, _ = self._model._predict(t_emb)

            # evaluate total variation distance
            arry_code_discrete = np.eye(n_ary)[t_code_discrete.data.numpy()]
            arry_code_continuous = t_code_continuous.data.numpy()
            tvd_b = total_variation_distance(arry_code_discrete, arry_code_continuous)
            # mean by digits, sum by samples
            tvd_sum += np.sum(np.mean(tvd_b, axis=-1))

        if average:
            return tvd_sum / n_eval
        else:
            return tvd_sum



class CodeCountEvaluator(object):

    def __init__(self, model: AutoEncoder, embeddings_dataset: ):
        pass


    def count_code_occurence(self):

        t_code = torch.tensor([])
        mat_code = t_code.numpy()
        mat_code_count = np.zeros((n_digits, n_ary), dtype=np.int)
        for digit, (vec_value, vec_count) in enumerate(map(lambda v: np.unique(v, return_counts=True), mat_code.T)):
            mat_code_count[digit, vec_value] += vec_count