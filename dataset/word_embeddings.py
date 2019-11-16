#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io, os
from typing import Union, Collection, Optional

from abc import ABCMeta, abstractmethod

from nltk.tokenize import MWETokenizer
import torch
from torch.utils.data import Dataset
import numpy as np
import fasttext
from gensim.models import KeyedVectors


class AbstractWordEmbeddingsDataset(Dataset, metaclass=ABCMeta):
    _PHRASE_DELIMITER = "_"
    _idx_to_word = {}
    transform = None

    def phrase_splitter(self, phrase: str):
        return self._mwe_tokenizer.tokenize(phrase.split(self._PHRASE_DELIMITER))

    def _init_mwe_tokenizer(self):
        def multi_word_expressions():
            for entity in self.vocab:
                if entity.find(self._PHRASE_DELIMITER) != -1:
                    yield entity.split(self._PHRASE_DELIMITER)
        it = multi_word_expressions()
        self._mwe_tokenizer = MWETokenizer(it)

    @abstractmethod
    def encode_phrase(self, phrase) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def encode(self, entity) -> Optional[np.ndarray]:
        pass

    def is_encodable(self, entity: str) -> bool:
        vec_e = self.encode(entity)
        return vec_e is not None

    def index_to_entity(self, index: int):
        return self._idx_to_word[index]

    @abstractmethod
    def vocab(self) -> Collection[str]:
        pass

    @abstractmethod
    def n_dim(self):
        pass

    def __getitem__(self, key: Union[torch.Tensor, int, str]):
        if torch.is_tensor(key):
            key = key.tolist()

        if isinstance(key, int):
            word = self._idx_to_word[key]
        elif isinstance(key, str):
            word = key
        else:
            raise NotImplementedError(f"unsupprted key type: {type(key)}")

        embedding = self.encode(word)
        assert embedding is not None, f"string `{word}` cannot be encoded."

        sample = {"entity":word, "embedding":embedding}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self._idx_to_word)


class GeneralPurposeEmbeddingsDataset(AbstractWordEmbeddingsDataset):

    def __init__(self, path_numpy_array_binary_format: str, path_vocabulary_text: str,
                 enable_phrase_composition:bool = False, transform=None):

        self.embedding = np.load(path_numpy_array_binary_format)
        sample_size = self.embedding.shape[0]
        self._vocabulary = self._load_vocabulary_text(path_vocabulary_text)
        assert len(self._vocabulary) == sample_size, "embeddings and vocabulary size mismatch detected."
        self._idx_to_word = {idx:word for idx, word in enumerate(self._vocabulary)}
        self._entity_to_idx = {word:idx for idx, word in enumerate(self._vocabulary)}
        self.transform = transform
        self._enable_phrase_composition = enable_phrase_composition
        if enable_phrase_composition:
            self._init_mwe_tokenizer()

    def _load_vocabulary_text(self, path_vocabulary_text: str):
        lst_v = []
        with io.open(path_vocabulary_text, mode="r") as ifs:
            for s in ifs:
                lst_v.append(s.strip())
        return lst_v

    def encode(self, entity):
        if entity in self.vocab:
            return self.embedding[self._entity_to_idx[entity], :]
        else:
            if self._enable_phrase_composition:
                return self.encode_phrase(entity)
            else:
                return None

    def encode_phrase(self, phrase):
        lst_tokens = self.phrase_splitter(phrase)
        if not all([token in self.vocab for token in lst_tokens]):
            return None
        vec_r = np.mean(np.stack([self.embedding[self._entity_to_idx[token],:] for token in lst_tokens]), axis=0)
        return vec_r

    @property
    def n_dim(self):
        return self.embedding.shape[1]

    @property
    def vocab(self):
        return self._entity_to_idx


class FastTextDataset(AbstractWordEmbeddingsDataset):

    def __init__(self, path_fasttext_binary_format: str, transform=None,
                 enable_phrase_composition=True):

        assert os.path.exists(path_fasttext_binary_format), f"file not found: {path_fasttext_binary_format}"
        self.model = fasttext.load_model(path_fasttext_binary_format)
        self.transform = transform
        self._enable_phrase_composition = enable_phrase_composition
        self._idx_to_word = {idx:word for idx, word in enumerate(self.model.get_words(on_unicode_error="ignore"))}
        self._vocab = set(self._idx_to_word.values())

        if enable_phrase_composition:
            self._init_mwe_tokenizer()

    def encode_phrase(self, phrase: str):
        lst_tokens = self.phrase_splitter(phrase)
        vec_r = np.mean(np.stack([self.model.get_word_vector(token) for token in lst_tokens]), axis=0)
        return vec_r

    def encode(self, entity: str):
        if entity in self.vocab:
            return self.model.get_word_vector(entity)
        else:
            if self._enable_phrase_composition:
                return self.encode_phrase(entity)
            else:
                return self.model.get_word_vector(entity)

    @property
    def n_dim(self):
        return self.model.get_dimension()

    @property
    def vocab(self):
        return self._vocab


class Word2VecDataset(AbstractWordEmbeddingsDataset):

    def __init__(self, path_word2vec_format: str, binary: bool = True, init_sims: bool = False, transform=None,
                 enable_phrase_composition=True, **kwargs):

        assert os.path.exists(path_word2vec_format), f"file not found: {path_word2vec_format}"
        self.model = KeyedVectors.load_word2vec_format(path_word2vec_format, binary=binary, **kwargs)
        if init_sims:
            self.model.init_sims(replace=True)
        self.transform = transform
        self._enable_phrase_composition = enable_phrase_composition
        self._idx_to_word = self.model.index2word

        if enable_phrase_composition:
            self._init_mwe_tokenizer()

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

    @property
    def n_dim(self):
        return self.model.vector_size

    @property
    def vocab(self):
        return self.model.vocab
