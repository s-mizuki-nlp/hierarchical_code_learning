#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import numpy as np

class NearestNeighborSearchEvaluator(object):

    def __init__(self, original_representation: np.ndarray, code_representaion: np.ndarray):

        assert original_representation.shape[0] == code_representaion.shape[0],\
        "sample size must be equal."

        self._mat_x = original_representation
        self._code_x = code_representaion
        self._n_sample = original_representation.shape[0]


    def precision_at_k(self, sample_index: int, topk: int = 10):

        assert sample_index < self._n_sample, "invalid sample index was specified."

        test_code = self._code_x[sample_index,]
        test_x = self._mat_x[sample_index,]

        # top-k similarity over code representation space
        ## remove itself
        vec_humming_distance = np.delete(np.count_nonzero(self._code_x - test_code, axis=-1), sample_index)
        topk_pred = np.argsort(vec_humming_distance)[:topk]

        # top-k similarity over original(=dense) representation space
        vec_l2_distance = np.delete(np.linalg.norm(self._mat_x - test_x, axis=-1), sample_index)
        topk_gt = np.argsort(vec_l2_distance)[:topk]

        # calculate precision@K
        prec_at_k = len(set(topk_pred) & set(topk_gt)) / topk

        return prec_at_k