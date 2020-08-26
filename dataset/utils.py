#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Iterable, Optional, Tuple, List
from collections import OrderedDict
import numpy as np
import math

# ToDo: implement methods
class EmbeddingSimilaritySearch(object):

    __EPS = 1E-7

    def __init__(self, embeddings: Dict[str, np.array]):
        self._embeddings = np.stack(embeddings.values())
        self._idx2entity = OrderedDict(enumerate(embeddings.keys()))
        self._entity2idx = OrderedDict(pair[::-1] for pair in self._idx2entity.items())

        # normalize
        self._embeddings = self._embeddings / (np.linalg.norm(self._embeddings, ord=2, axis=1, keepdims=True) + self.__EPS)

    @property
    def vocab(self):
        return self._idx2entity.values()

    @property
    def n_vocab(self):
        return len(self.vocab)

    @property
    def n_dim(self):
        return self._embeddings.shape[1]

    def most_similar(self, vector: np.array, top_k: Optional[int] = None, top_q: Optional[float] = None, excludes: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        assert (top_k is not None) or (top_q is not None), "you must specify either `top_k` or `top_q` argument."

        if top_q is not None:
            top_k = math.ceil(self.n_vocab * top_q)
        assert top_k > 0, "`top_k` must be greater than zero."

        if excludes is None:
            idx2entity = self._idx2entity
            embeddings = self._embeddings
        else:
            entities = tuple(entity for entity in self._idx2entity.values() if entity not in excludes)
            if len(entities) == 0:
                return []

            indices = list(map(self._entity2idx.get, entities))
            idx2entity = OrderedDict(enumerate(entities))
            embeddings = self._embeddings[indices,:]

        return self._most_similar_topk(vector, embeddings, idx2entity, top_k)

    def _most_similar_topk(self, vector: np.array, embeddings: np.ndarray, idx2entity: Dict[int, str], top_k: int) -> List[Tuple[str, float]]:
        vector = vector / (np.linalg.norm(vector, ord=2) + self.__EPS)

        vec_similarity = embeddings.dot(vector)
        vec_indices_topk = np.argsort(-vec_similarity)[:top_k]

        lst_ret = [(idx2entity[idx], vec_similarity[idx]) for idx in vec_indices_topk]
        return lst_ret