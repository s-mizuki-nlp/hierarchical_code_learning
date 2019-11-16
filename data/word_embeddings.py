#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
from typing import Union

from nltk.tokenize import MWETokenizer
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


class GeneralPurposeEmbeddingsDataset(Dataset):

    def __init__(self, path_numpy_array_binary_format: str, path_vocabulary_text: str, transform=None):

        self.embedding = np.load(path_numpy_array_binary_format)
        sample_size = self.embedding.shape[0]
        self._vocabulary = self._load_vocabulary_text(path_vocabulary_text)
        assert len(self._vocabulary) == sample_size, "embeddings and vocabulary size mismatch detected."
        self._idx_to_word = {idx:word for idx, word in enumerate(self._vocabulary)}
        self.transform = transform
        self._n_sample = sample_size

    def _load_vocabulary_text(self, path_vocabulary_text: str):
        lst_v = []
        with io.open(path_vocabulary_text, mode="r") as ifs:
            for s in ifs:
                lst_v.append(s.strip())
        return lst_v

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

    _PHRASE_DELIMITER = "_"

    def __init__(self, path_word2vec_format: str, binary: bool = True, init_sims: bool = False, transform=None,
                 enable_phrase_composition=True, **kwargs):

        assert os.path.exists(path_word2vec_format), f"file not found: {path_word2vec_format}"
        self.model = KeyedVectors.load_word2vec_format(path_word2vec_format, binary=binary, **kwargs)
        if init_sims:
            self.model.init_sims(replace=True)
        self.transform = transform
        self._enable_phrase_composition = enable_phrase_composition
        self._idx_to_word = self.model.index2word
        self._n_sample = len(self.model.vocab)

        if enable_phrase_composition:
            self._init_mwe_tokenizer()

    def _init_mwe_tokenizer(self):
        def multi_word_expressions():
            for entity in self.vocab:
                if entity.find(self._PHRASE_DELIMITER) != -1:
                    yield entity.split(self._PHRASE_DELIMITER)
        it = multi_word_expressions()
        self._mwe_tokenizer = MWETokenizer(it)

    def __len__(self):
        return self._n_sample

    def __getitem__(self, key: Union[torch.Tensor, int, str]):

        if torch.is_tensor(key):
            key = key.tolist()

        if isinstance(key, int):
            word = self._idx_to_word[key]
            embedding = self.model.get_vector(word)
        elif isinstance(key, str):
            word = key
            embedding = self.encode(word)
            assert embedding is not None, f"string {word} cannot be encoded."

        sample = {"entity":word, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def phrase_splitter(self, phrase: str):
        return self._mwe_tokenizer.tokenize(phrase.split(self._PHRASE_DELIMITER))

    def encode_phrase(self, phrase: str):
        lst_tokens = self.phrase_splitter(phrase)
        if not all([token in self.vocab for token in lst_tokens]):
            return None
        vec_r = np.mean(np.stack([self.model.get_vector(token) for token in lst_tokens]), axis=0)
        return vec_r

    def encode(self, entity: str):
        if entity in self.vocab:
            return self.model.get_vector(entity)
        else:
            if self._enable_phrase_composition:
                return self.encode_phrase(entity)
            else:
                return None

    def is_encodable(self, entity: str):
        vec_e = self.encode(entity)
        return vec_e is not None

    @property
    def n_dim(self):
        return self.model.vector_size

    @property
    def vocab(self):
        return self.model.vocab


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
