#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional
import numpy as np
import pydash
import torch
from torch.utils.data import DataLoader
from dataset.word_embeddings import AbstractWordEmbeddingsDataset
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

    def __init__(self, model: AutoEncoder, embeddings_dataset: AbstractWordEmbeddingsDataset,
                 **kwargs_dataloader):
        self._n_digits, self._n_ary = model.n_digits, model.n_ary
        self._model = model
        self._embeddings_dataset = embeddings_dataset
        self._embeddings_data_loader = DataLoader(embeddings_dataset, **kwargs_dataloader)

    def evaluate(self,
                 embedding_key_name: str = "embedding",
                 ground_truth_key_path: Optional[str] = "entity_info.code_representation",
                 normalize: bool = True):

        if ground_truth_key_path is not None:
            use_ground_truth = True

        else:
            use_ground_truth = False

        mat_code_count_pred = np.zeros((self._n_digits, self._n_ary), dtype=np.int)
        mat_code_count_gt = mat_code_count_pred.copy() if use_ground_truth else None

        for batch in self._embeddings_data_loader:
            t_x = pydash.objects.get(batch, embedding_key_name)
            # mat_code: (n_batch, n_digits)
            mat_code = self._model._encode(t_x, adjust_code_probability=True).numpy()

            # count value occurence at each digit of the code.
            for digit, (vec_value, vec_count) in enumerate(map(lambda v: np.unique(v, return_counts=True), mat_code.T)):
                mat_code_count_pred[digit, vec_value] += vec_count

            # similarly, count value occurence of the ground-truth code if specified
            if use_ground_truth:
                # mat_code_gt: (n_batch, n_digits)
                mat_code_gt = np.stack(pydash.objects.get(batch, ground_truth_key_path)).T
                for digit, (vec_value, vec_count) in enumerate(map(lambda v: np.unique(v, return_counts=True), mat_code_gt.T)):
                    mat_code_count_gt[digit, vec_value] += vec_count

        if normalize:
            n_sample = len(self._embeddings_dataset)
            mat_code_count_pred = mat_code_count_pred / n_sample
            if use_ground_truth:
                mat_code_count_gt = mat_code_count_gt / n_sample

        return mat_code_count_pred, mat_code_count_gt