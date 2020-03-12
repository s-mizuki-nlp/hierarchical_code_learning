#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union
from collections import defaultdict
import numpy as np
import math
from functools import lru_cache

from .lexical_knowledge import HyponymyDataset


class BasicHyponymyPairSet(object):

    _DEBUG_MODE = False

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as a DAG
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset)
        self.record_ancestors_and_descendants(iter_hyponymy_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    @property
    def nodes(self):
        return self._nodes

    @property
    def n_nodes_max(self):
        return len(self.nodes)

    @property
    def trainset_ancestors(self):
        return self._trainset_ancestors

    @property
    def trainset_descendants(self):
        return self._trainset_descendants

    def record_ancestors_and_descendants(self, iter_hyponymy_pairs):
        self._trainset_ancestors = defaultdict(set)
        self._trainset_descendants = defaultdict(set)
        for hypernym, hyponym in iter_hyponymy_pairs:
            self._trainset_ancestors[hyponym].add(hypernym)
            self._trainset_descendants[hypernym].add(hyponym)
        self._nodes = tuple(set(self._trainset_ancestors.keys()) | set(self._trainset_descendants.keys()))

    def _random_number_generator_iterator(self, max_value: int):
        seeds = np.arange(max_value)
        while True:
            np.random.shuffle(seeds)
            for idx in seeds:
                yield idx

    def hypernyms(self, entity):
        return self.trainset_ancestors.get(entity, set())

    def hyponyms(self, entity):
        return self.trainset_descendants.get(entity, set())

    @lru_cache(maxsize=100000)
    def hypernyms_and_hyponyms_and_self(self, entity):
        return self.hyponyms(entity) | self.hypernyms(entity) | {entity}

    def hyponyms_and_self(self, entity):
        return self.hyponyms(entity) | {entity}

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True) -> List[str]:

        if exclude_hypernyms:
            non_candidates = self.hypernyms_and_hyponyms_and_self(entity)
        else:
            non_candidates = self.hyponyms_and_self(entity)
        candidates = self.nodes if candidates is None else tuple(set(candidates).intersection(set(self.nodes)))

        if len(candidates) - len(non_candidates) <= 0:
            return []
        elif len(non_candidates)/len(candidates) >= 0.9:
            candidates = tuple(set(candidates) - non_candidates)
        elif len(candidates) < size:
            candidates = (candidates)*(math.ceil(size/len(candidates)))

        # sampling with replacement
        sampled = tuple()
        n_candidates = len(candidates)
        for rnd_idx in self._random_number_generator:
            sampled_new = candidates[rnd_idx % n_candidates]
            if sampled_new in non_candidates:
                continue
            sampled = sampled + (sampled_new,)
            if len(sampled) >= size:
                break

        return sampled

    def sample_random_hyponyms(self, entity: str,
                               candidates: Optional[Iterable[str]] = None,
                               size: int = 1, exclude_hypernyms: bool = True, **kwargs):
        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms)
        lst_ret = [(entity, hyponym, -1.0) for hyponym in lst_non_hyponymy_entities]

        return lst_ret

    def sample_random_hypernyms(self, entity: str,
                                candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True, **kwargs):

        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms)
        lst_ret = [(hypernym, entity, -1.0) for hypernym in lst_non_hyponymy_entities]

        return lst_ret

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None, **kwargs):
        if (hypernym not in self.nodes) or (hyponym not in self.nodes):
            return not_exists

        if include_reverse_hyponymy:
            candidates = self.hyponyms(hypernym) | self.hypernyms(hypernym)
        else:
            candidates = self.hyponyms(hypernym)
        ret = hyponym in candidates

        return ret