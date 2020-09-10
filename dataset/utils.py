#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Iterable, Optional, Tuple, List, Union
import warnings
from collections import OrderedDict
try:
    import cupy as xp
    IS_CUPY_AVAILABLE = True
except:
    warnings.warn("cuda device is unavailable. we will use numpy instead.")
    import numpy as xp
    IS_CUPY_AVAILABLE = False

import math

class EmbeddingSimilaritySearch(object):

    __EPS = 1E-7
    __CUDA_DEVICE_ID = 0

    def __init__(self, embeddings: Dict[str, xp.array]):
        self._embeddings = xp.stack([xp.array(v) for v in embeddings.values()])
        self._idx2entity = OrderedDict(enumerate(embeddings.keys()))
        self._entity2idx = OrderedDict(pair[::-1] for pair in self._idx2entity.items())

        # normalize
        self._embeddings = self._embeddings / (xp.linalg.norm(self._embeddings, ord=2, axis=1, keepdims=True) + self.__EPS)

    @classmethod
    def SET_CUDA_DEVICE_ID(cls, device_id: int):
        if IS_CUPY_AVAILABLE:
            xp.cuda.Device(device_id).use()

    @property
    def vocab(self):
        return self._idx2entity.values()

    @property
    def n_vocab(self):
        return len(self.vocab)

    @property
    def n_dim(self):
        return self._embeddings.shape[1]

    def most_similar(self, entity: Optional[str] = None, vector: Optional[xp.array] = None,
                     top_k: Optional[int] = None, top_q: Optional[float] = None, excludes: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:

        assert (entity is not None) or (vector is not None), "you must specify either `entity` or `vector` argument."
        assert (top_k is not None) or (top_q is not None), "you must specify either `top_k` or `top_q` argument."

        if entity is not None:
            assert entity in self._entity2idx, f"'{entity}' is not found."
            vector = self._embeddings[self._entity2idx[entity]]
            remove_query_entity = True
        else:
            vector = xp.asarray(vector)
            remove_query_entity = False

        if top_q is not None:
            top_k = math.ceil(self.n_vocab * top_q)
        assert top_k > 0, "`top_k` must be greater than zero."

        if excludes is None:
            idx2entity = self._idx2entity
            embeddings = self._embeddings
        else:
            out_of_candidate_entities = tuple(entity for entity in self._idx2entity.values() if entity not in excludes)
            if len(out_of_candidate_entities) == 0:
                return []

            indices = list(map(self._entity2idx.get, out_of_candidate_entities))
            idx2entity = OrderedDict(enumerate(out_of_candidate_entities))
            embeddings = self._embeddings[indices,:]

        return self._most_similar_topk(vector, embeddings, idx2entity, top_k, remove_query_entity)

    def _most_similar_topk(self, vector: xp.array, embeddings: xp.ndarray, idx2entity: Dict[int, str], top_k: int, remove_query_entity: bool = False) -> List[Tuple[str, float]]:
        vector = vector / (xp.linalg.norm(vector, ord=2) + self.__EPS)

        vec_similarity = xp.dot(embeddings, vector)
        if remove_query_entity:
            vec_indices_topk = xp.argsort(-vec_similarity)[1:top_k + 1]
        else:
            vec_indices_topk = xp.argsort(-vec_similarity)[:top_k]
        if hasattr(xp, "asnumpy"):
            # convert from cupy object to numpy array.
            vec_indices_topk = xp.asnumpy(vec_indices_topk)
            vec_similarity = xp.asnumpy(vec_similarity)

        lst_ret = [(idx2entity[idx], vec_similarity[idx]) for idx in vec_indices_topk]
        return lst_ret
