#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Iterable, Optional, Tuple, List
import numpy as np

# ToDo: implement methods
class EmbeddingSimilaritySearch(object):

    def __init__(self, embeddings: Dict[str, np.array]):
        pass

    def most_similar(self, vector: np.array, top_k: Optional[int] = None, top_q: Optional[float] = None, excludes: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        pass

    def _most_similar_topk(self, vector: np.array, top_k: int, embeddings: np.ndarray, id2entity: Dict[int, str]) -> List[Tuple[str, float]]:
        pass