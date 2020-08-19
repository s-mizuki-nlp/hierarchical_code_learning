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
        raise NotImplementedError("invalid method.")

    @property
    def trainset_descendants(self):
        raise NotImplementedError("invalid method.")

    @property
    def trainset_hyponymies_and_self(self):
        return self._trainset_hyponymies_and_self

    @property
    def trainset(self):
        return self._trainset

    def record_ancestors_and_descendants(self, iter_hyponymy_pairs):
        self._trainset  = set() # set of the tuple of (hypernym, hyponym) pair.
        self._trainset_hyponymies_and_self = defaultdict(set)
        for hypernym, hyponym in iter_hyponymy_pairs:
            self._trainset_hyponymies_and_self[hyponym].add(hypernym)
            self._trainset_hyponymies_and_self[hypernym].add(hyponym)
            self._trainset.add((hypernym, hyponym))
        # insert oneself
        for entity in self._trainset_hyponymies_and_self.keys():
            self._trainset_hyponymies_and_self[entity].add(entity)

        self._nodes = tuple(set(self._trainset_hyponymies_and_self.keys()))

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

    def hyponyms_and_self(self, entity):
        return self.hyponyms(entity) | {entity}

    def hypernyms_and_hyponyms_and_self(self, entity):
        return self.trainset_hyponymies_and_self.get(entity, set())

    def hyponymies(self):
        return self.trainset

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True) -> List[str]:

        if exclude_hypernyms:
            non_candidates = self.hypernyms_and_hyponyms_and_self(entity)
        else:
            non_candidates = self.hyponyms_and_self(entity)
        candidates = self.nodes if candidates is None else tuple(set(candidates).intersection(set(self.nodes)))

        non_candidate_ratio = len(non_candidates)/len(candidates)
        if 0.999 <= non_candidate_ratio:
            # give up sampling
            return []
        elif 0.9 <= non_candidate_ratio < 0.999:
            candidates = tuple(set(candidates) - non_candidates)

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

    def is_in_dataset(self, entity, **kwargs):
        return entity in self.trainset_hyponymies_and_self

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None, **kwargs):
        if not (self.is_in_dataset(hypernym) and self.is_in_dataset(hyponym)):
            return not_exists

        pair_forward = (hypernym, hyponym)
        pair_reverse = (hyponym, hypernym)

        if include_reverse_hyponymy:
            ret = (pair_forward in self.hyponymies()) or (pair_reverse in self.hyponymies())
        else:
            ret = (pair_forward in self.hyponymies())

        return ret


class WordNetHyponymyPairSet(BasicHyponymyPairSet):

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as for each part-of-speech tags as DAG
        dict_iter_trainset_pairs = defaultdict(list)
        for record in hyponymy_dataset:
            entity_type = record["pos"]
            entity_hyper = record["hypernym"]
            entity_hypo = record["hyponym"]
            dict_iter_trainset_pairs[entity_type].append((entity_hyper, entity_hypo))

        self.record_ancestors_and_descendants(dict_iter_trainset_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    def record_ancestors_and_descendants(self, dict_iter_hyponymy_pairs):
        self._trainset = defaultdict(lambda :set())
        self._trainset_hyponymies_and_self = defaultdict(lambda :defaultdict(set))

        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            for hypernym, hyponym in iter_hyponymy_pairs:
                self._trainset[entity_type].add((hypernym, hyponym))
                self._trainset_hyponymies_and_self[entity_type][hyponym].add(hypernym)
                self._trainset_hyponymies_and_self[entity_type][hypernym].add(hyponym)
            # add oneself
            for entity in self._trainset_hyponymies_and_self.keys():
                self._trainset_hyponymies_and_self[entity_type][entity].add(entity)

        # create tuple of entities as nodes
        nodes = {}
        for entity_type in dict_iter_hyponymy_pairs.keys():
            nodes[entity_type] = tuple(set(self._trainset_hyponymies_and_self[entity_type].keys()))
        self._nodes = nodes

        # unset active entity type
        self._active_entity_type = None

    @property
    def ACTIVE_ENTITY_TYPE(self):
        return self._active_entity_type

    @ACTIVE_ENTITY_TYPE.setter
    def ACTIVE_ENTITY_TYPE(self, value):
        if value is not None:
            self._active_entity_type = value

    @property
    def entity_types(self):
        return set(self._trainset.keys())

    @property
    def trainset_ancestors(self):
        raise NotImplementedError("deprecated method.")

    @property
    def trainset_descendants(self):
        raise NotImplementedError("deprecated method.")

    @property
    def nodes(self):
        return self._nodes.get(self.ACTIVE_ENTITY_TYPE, self._nodes)

    @property
    def trainset_hyponymies_and_self(self):
        return self._trainset_hyponymies_and_self.get(self.ACTIVE_ENTITY_TYPE, self._trainset_hyponymies_and_self)

    @property
    def trainset(self):
        return self._trainset.get(self.ACTIVE_ENTITY_TYPE, self._trainset)

    @property
    def n_nodes_max(self):
        return max(map(len, self._nodes.values()))

    def sample_random_hypernyms(self, entity: str, candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().sample_random_hypernyms(entity, candidates, size, exclude_hypernyms)

    def sample_random_hyponyms(self, entity: str, candidates: Optional[Iterable[str]] = None,
                               size: int = 1, exclude_hypernyms: bool = True, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().sample_random_hyponyms(entity, candidates, size, exclude_hypernyms)

    def is_in_dataset(self, entity, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().is_in_dataset(entity)

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().is_hyponymy_relation(hypernym, hyponym, include_reverse_hyponymy, not_exists)