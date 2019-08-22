#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os

import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
import fasttext
from gensim.models import KeyedVectors
from wikipedia2vec import Wikipedia2Vec

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


class FastTextDataset(Dataset):

    def __init__(self, path_fasttext_binary_format: str, transform=None):

        assert os.path.exists(path_fasttext_binary_format), f"file not found: {path_fasttext_binary_format}"
        self.model = fasttext.load_model(path_fasttext_binary_format)
        self.transform = transform
        self._idx_to_word = {idx:word for idx, word in enumerate(self.model.get_words(on_unicode_error="ignore"))}
        self._n_sample = len(self._idx_to_word)

    def __len__(self):
        return self._n_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        word = self._idx_to_word[idx]
        embedding = self.model.get_word_vector(word)

        sample = {"entity":word, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def n_dim(self):
        return self.model.get_dimension()


class Word2VecDataset(Dataset):

    def __init__(self, path_word2vec_format: str, binary: bool = True, init_sims: bool = False, transform=None, **kwargs):

        assert os.path.exists(path_word2vec_format), f"file not found: {path_word2vec_format}"
        self.model = KeyedVectors.load_word2vec_format(path_word2vec_format, binary=binary, **kwargs)
        if init_sims:
            self.model.init_sims(replace=True)
        self.transform = transform
        self._idx_to_word = self.model.index2word
        self._n_sample = len(self.model.vocab)

    def __len__(self):
        return self._n_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        word = self._idx_to_word[idx]
        embedding = self.model.get_vector(word)

        sample = {"entity":word, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @property
    def n_dim(self):
        return self.model.vector_size


class Wikipedia2VecDataset(Dataset):

    def __init__(self, path_wikipedia2vec: str, transform=None):

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
