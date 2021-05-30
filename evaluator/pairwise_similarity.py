#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import copy
from typing import Dict, List, Tuple
import math, random
import warnings
from itertools import combinations, product

from nltk.corpus import wordnet as wn
import numpy as np

from dataset.word_embeddings import GeneralPurposeEmbeddingsDataset


class PairwiseSimilarityEvaluator(object):

    def __init__(self, dict_cluster_ids: Dict[str, List[str]], part_of_speech: str,
                 dataset_embeddings: GeneralPurposeEmbeddingsDataset,
                 cluster_type: str = None):
        self._dict_cluster_ids = dict_cluster_ids
        self._pos = part_of_speech
        self._dataset_embeddings = dataset_embeddings

        if cluster_type is not None:
            self._assert_pos_uniqueness(cluster_type)

    def _assert_pos_uniqueness(self, cluster_type: str):
        ## extract part-of-speech from keys
        if cluster_type == "synset":
            id_to_pos = _cluster_id_synset_to_pos
        elif cluster_type == "code":
            id_to_pos = _cluster_id_code_to_pos
        else:
            raise ValueError(f"unknown cluster type: {cluster_type}")

        # assertion
        lst_pos = [id_to_pos(cluster_id) for cluster_id in self._dict_cluster_ids.keys()]
        set_pos = set(lst_pos)
        if len(set_pos) > 1:
            raise ValueError(f"you can't pass mixed part-of-speech cluster: {set_pos}")

    def get_num_total_word_pairs(self, word_pair_type: str):
        if word_pair_type in ("in-cluster", "in_cluster"):
            return self._num_in_cluster_word_pairs
        elif word_pair_type in ("between-cluster", "between_cluster"):
            return self._num_between_cluster_word_pairs
        else:
            raise ValueError(f"unknown word_pair_type: {word_pair_type}")

    @property
    def _num_in_cluster_word_pairs(self):
        return count_total_in_cluster_pairs(self._dict_cluster_ids)

    @property
    def _num_between_cluster_word_pairs(self):
        return count_total_between_cluster_pairs(self._dict_cluster_ids)

    def wn_1st_sense_wu_palmer_similarity(self, lemma_1: str, lemma_2: str):
        return wn_1st_sense_wu_palmer_similarity(self._pos, lemma_1, lemma_2)

    def cosine_similarity(self, lemma_1: str, lemma_2: str):
        return self._dataset_embeddings.cosine_similarity(lemma_1, lemma_2)

    def batch_wn_1st_sense_wu_palmer_similarity(self, lst_lemma_pairs: List[Tuple[str, str]]):
        return [self.wn_1st_sense_wu_palmer_similarity(*lemma_pair) for lemma_pair in lst_lemma_pairs]

    def batch_cosine_similarity(self, lst_lemma_pairs: List[Tuple[str, str]]):
        return [self.cosine_similarity(*lemma_pair) for lemma_pair in lst_lemma_pairs]

    def _sample_in_cluster_word_pairs(self, size: int):
        n_size_max = self._num_in_cluster_word_pairs
        if size >= n_size_max:
            warnings.warn(f"sample size is reduced to: {n_size_max}")
            size = n_size_max

        lst_ret = sample_in_cluster_word_pairs(dict_cluster_ids=self._dict_cluster_ids, size=size)
        return lst_ret

    def _sample_between_cluster_word_pairs(self, size: int):
        n_size_max = self._num_between_cluster_word_pairs
        if size >= n_size_max:
            warnings.warn(f"sample size is reduced to: {n_size_max}")
            size = n_size_max

        lst_ret = sample_between_cluster_word_pairs(dict_cluster_ids=self._dict_cluster_ids, size=size)
        return lst_ret

    def sample_word_pairs(self, size: int, word_pair_type: str):
        if word_pair_type in ("in-cluster", "in_cluster"):
            return self._sample_in_cluster_word_pairs(size=size)
        elif word_pair_type in ("between-cluster", "between_cluster"):
            return self._sample_between_cluster_word_pairs(size=size)
        else:
            raise ValueError(f"unknown word_pair_type: {word_pair_type}")


def wn_1st_sense_wu_palmer_similarity(pos: str, lemma_1: str, lemma_2: str):
    synset_1 = wn.synsets(lemma_1, pos=pos)[0]
    synset_2 = wn.synsets(lemma_2, pos=pos)[0]
    if synset_1 is synset_2:
        return 1.0
    else:
        return wn.wup_similarity(synset_1, synset_2)


def count_total_in_cluster_pairs(dict_cluster_ids):
    # クラスタ内ペア数を計算
    lst_elements = [len(lst) for lst in dict_cluster_ids.values()]
    it_pairs = map(_calc_pair_size, lst_elements)
    return sum(it_pairs)

def count_total_between_cluster_pairs(dict_cluster_ids):
    # 可能なクラスタ間ペア数を計算
    it_samples = map(len, dict_cluster_ids.values())
    return sum(it_samples)

def _calc_pair_size(n):
    # ペア数を計算
    if n <= 1:
        return 0
    else:
        return (n * (n-1)) // 2

def _allocate_total_size(lst_max_sizes, total_size: int):
    # total_size個のサンプルをN個のクラスタに配分．ただし各クラスタの上限はn_tとする．
    n_max_total = sum(lst_max_sizes)
    assert total_size <= n_max_total, f"total_size must be smaller than: {n_max_total}"
    if n_max_total == total_size:
        return lst_max_sizes

    ratio = total_size / n_max_total
    lst_n_hat = [math.floor(n*ratio) for n in lst_max_sizes]
    lst_n_delta = [math.ceil(n*ratio - n_hat) for n, n_hat in zip(lst_max_sizes, lst_n_hat)]

    delta = total_size - sum(lst_n_hat)
    for t, n_delta in enumerate(lst_n_delta):
        lst_n_hat[t] += n_delta
        delta -= n_delta
        if delta == 0:
            break

    assert sum(lst_n_hat) == total_size, f"something went wrong: {sum(lst_n_hat)} != {total_size}"
    for n_hat , n_max in zip(lst_n_hat, lst_max_sizes):
        assert n_hat <= n_max, f"something went wrong: {sum(n_hat)} > {n_max}"

    return lst_n_hat


def _sample_lemma_pairs(lst_lemmas, size):
    n_lemmas = len(lst_lemmas)
    size_max = _calc_pair_size(n_lemmas)
    if size == 0:
        return []
    elif size == 1:
        return [tuple(np.random.permutation(lst_lemmas)[:2])]
    elif size == size_max:
        return list(combinations(lst_lemmas, 2))
    elif size < size_max:
        lst_ = list(combinations(lst_lemmas, 2))
        random.shuffle(lst_)
        return lst_[:size]
    else:
        raise ValueError(f"`size` must be smaller than: {size_max}")

def _sample_lemma_pairs_from_two_lists(lst_lemmas_x, lst_lemmas_y, size: int):
    size_max = len(lst_lemmas_x) * len(lst_lemmas_y)
    if size == 0:
        return []
    elif size == size_max:
        return list(product(lst_lemmas_x, lst_lemmas_y))
    elif size < size_max:
        lst_ = list(product(lst_lemmas_x, lst_lemmas_y))
        random.shuffle(lst_)
        return lst_[:size]
    else:
        raise ValueError(f"`size` must be smaller than: {size_max}")

def _cluster_id_code_to_pos(cluster_id: str):
    return cluster_id[0]

def _cluster_id_synset_to_pos(cluster_id: str):
    return wn.synset(cluster_id).pos()

def sample_in_cluster_word_pairs(dict_cluster_ids: Dict[str, List[str]], size: int):
    # クラスタ内単語ペアをサンプリング．
    # return: sampled pairs; [(lemma_x1, lemma_y1),...]

    # extract list of lemmas
    lst_lst_lemmas = [lst for lst in dict_cluster_ids.values()]

    # compute total number of pairs
    lst_max_sizes = list(map(_calc_pair_size, map(len, lst_lst_lemmas)))
    # allocate sample sizes
    lst_sizes = _allocate_total_size(lst_max_sizes=lst_max_sizes, total_size=size)
    # sample within-cluster lemma pairs
    lst_ret = []
    for lst_lemmas, size in zip(lst_lst_lemmas, lst_sizes):
        if size == 0:
            continue
        lst_tup_lemma_pairs = _sample_lemma_pairs(lst_lemmas, size)
        lst_ret.extend(lst_tup_lemma_pairs)
    return lst_ret


def sample_between_cluster_word_pairs(dict_cluster_ids: Dict[str, List[str]], size: int):
    # クラスタ間単語ペアをサンプリング．
    # return: sampled pairs; [(lemma_x1, lemma_y1),...]

    # extract cluster ids, then shuffle it
    lst_cluster_ids = list(dict_cluster_ids.keys())
    random.shuffle(lst_cluster_ids)
    # extract lemmas
    lst_lst_lemmas = [dict_cluster_ids[cluster_id] for cluster_id in lst_cluster_ids]
    lst_n_lemmas = list(map(len, lst_lst_lemmas))

    # compute available number of pairs
    lst_max_sizes = [n*n_next for n, n_next in zip(lst_n_lemmas[:-1], lst_n_lemmas[1:])] + [0,]

    # allocate sample sizes
    n_size_max = sum(lst_max_sizes)
    if size >= n_size_max:
        warnings.warn(f"sample size is reduced to: {n_size_max}")
        size = n_size_max
    lst_sizes = _allocate_total_size(lst_max_sizes=lst_max_sizes, total_size=size)

    # return between-cluster lemma pairs: [(lemma_x, lemma_y)] format

    lst_ret = []
    for cluster_id, cluster_id_next, size in zip(lst_cluster_ids[:-1], lst_cluster_ids[1:], lst_sizes):
        if size == 0:
            continue
        lst_lemmas = dict_cluster_ids[cluster_id]
        lst_lemmas_next = dict_cluster_ids[cluster_id_next]
        lst_tup_lemma_pairs = _sample_lemma_pairs_from_two_lists(lst_lemmas, lst_lemmas_next, size)
        lst_ret.extend(lst_tup_lemma_pairs)

    return lst_ret
