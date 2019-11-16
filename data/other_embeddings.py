#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from wikipedia2vec.wikipedia2vec import Wikipedia2Vec


class ToyEmbeddingsDataset(Dataset):

    def __init__(self, sample_size: int, embedding_dim: int, transform=None, seed: int = 0):

        np.random.seed(seed)
        self.embedding = np.random.normal(size=sample_size * embedding_dim).astype(np.float32).reshape((sample_size, embedding_dim))
        self._idx_to_word = {idx:f"{idx}" for idx in range(sample_size)}
        self.transform = transform
        self._n_sample = sample_size

    def __len__(self):
        return self._n_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        word = self._idx_to_word[idx]
        embedding = self.embedding[idx,:]

        sample = {"entity":word, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def n_dim(self):
        return self.embedding.shape[1]


class Wikipedia2VecDataset(Dataset):

    def __init__(self, path_wikipedia2vec: str, transform=None):
        warnings.warn("experimental dataset.")

        assert os.path.exists(path_wikipedia2vec), f"file not found: {path_wikipedia2vec}"
        self.model = Wikipedia2Vec.load(path_wikipedia2vec)
        self.transform = transform
        self._idx_to_entity = {idx:entity for idx, entity in enumerate(self.model.dictionary.entities())}
        self._n_sample = len(self._idx_to_entity)

    def __len__(self):
        return self._n_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        entity = self._idx_to_entity[idx]
        embedding = self.model.get_vector(entity)

        sample = {"entity":entity.title, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def n_dim(self):
        return self.model.train_params["dim_size"]