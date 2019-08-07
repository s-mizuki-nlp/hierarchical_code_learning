#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import math
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Dict, Tuple, Union, Any, Optional

from more_itertools import chunked


class AbstractFeeder(object):

    __metaclass__ = ABCMeta

    def __init__(self, n_minibatch=1):
        self._n_mb = n_minibatch

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def _data_iterator(self, *args, **kwargs) -> Iterable:
        pass

    def __iter__(self):
        # default iterator
        iter_data = self._data_iterator()
        iter_batch = chunked(iter_data, n=self._n_mb)
        for lst_batch in iter_batch:
            yield lst_batch


class EmbeddingsFeeder(AbstractFeeder):

    def __init__(self, mat_embeddings: np.ndarray, n_minibatch: int = 1):

        super(AbstractFeeder, self).__init__(n_minibatch)
        self._embeddings = mat_embeddings

    def __len__(self):
        return self._embeddings.shape[0]

    def _data_iterator(self) -> Iterable:
        pass

    def __iter__(self):
        n_splits = int(len(self) // self._n_mb)
        iter_batch = np.array_split(self._embeddings, n_splits, axis=0)
        for batch in iter_batch:
            yield batch

