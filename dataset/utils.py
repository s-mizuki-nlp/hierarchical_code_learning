#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable
from collections import defaultdict, Counter
from functools import lru_cache
import warnings
import networkx as nx
import numpy as np
import progressbar

from .lexical_knowledge import HyponymyDataset

class BasicTaxonomy(object):

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as a DAG
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset if record["distance"] == 1.0)
        self.build_directed_acyclic_graph(iter_hyponymy_pairs)
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset)
        self.record_ancestors_and_descendeants(iter_hyponymy_pairs)

    @property
    def dag(self):
        return self._dag

    @property
    def nodes(self):
        return self._nodes

    def build_directed_acyclic_graph(self, iter_hyponymy_pairs: Iterable[Tuple[str, str]]):
        """
        build taxonomy as a DAG based on the set of hyponymy relations

        @param iter_hyponymy_pairs: iterable of the tuple (hypernym, hyponym)
        """
        graph = nx.DiGraph()
        graph.add_edges_from(iter_hyponymy_pairs)

        self._dag = graph
        self._cache_root_nodes = {}
        self._nodes = set(graph.nodes)

    def record_ancestors_and_descendeants(self, iter_hyponymy_pairs):
        self._hyponym_frequency = Counter()
        self._hypernym_frequency = Counter()
        self._ancestors = defaultdict(set)
        self._descendants = defaultdict(set)
        for hypernym, hyponym in iter_hyponymy_pairs:
            self._ancestors[hyponym].add(hypernym)
            self._descendants[hypernym].add(hyponym)
            self._hyponym_frequency[hyponym] += 1
            self._hypernym_frequency[hypernym] += 1

    def _find_root_nodes(self, graph) -> Set[str]:
        hash_value = graph.__hash__()
        if hash_value in self._cache_root_nodes:
            return self._cache_root_nodes[hash_value]

        root_nodes = set([k for k,v in graph.in_degree() if v == 0])
        self._cache_root_nodes[hash_value] = root_nodes
        return root_nodes

    def hyponym_frequency(self, entity, not_exist: int = 0):
        return self._hyponym_frequency.get(entity, not_exist)

    def hypernym_frequency(self, entity, not_exist: int = 0):
        return self._hypernym_frequency.get(entity, not_exist)

    def hypernyms(self, entity):
        return nx.ancestors(self.dag, entity).union(self._ancestors.get(entity, set()))

    def hyponyms(self, entity):
        return nx.descendants(self.dag, entity).union(self._descendants.get(entity, set()))

    @lru_cache(maxsize=1000)
    def ancestors_and_descendents(self, entity):
        return self.hyponyms(entity) | self.hypernyms(entity) | {entity}

    @lru_cache(maxsize=1000)
    def descendents(self, entity):
        return self.hyponyms(entity) | {entity}

    def depth(self, entity, offset=1, not_exists=None):
        graph = self.dag

        if entity not in graph:
            return not_exists

        direct_root_nodes = nx.ancestors(graph, entity) & self._find_root_nodes(graph)
        if len(direct_root_nodes) == 0:
            depth = 0
        else:
            f_path_length_to_entity = lambda source: nx.shortest_path_length(graph, source, entity)
            depth = max(map(f_path_length_to_entity, direct_root_nodes))

        return depth + offset

    def hyponymy_distance(self, hypernym, hyponym, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        lowest_common_ancestor = nx.lowest_common_ancestor(graph, hypernym, hyponym)
        # 1) not connected
        if lowest_common_ancestor is None:
            dist = - self.depth(hypernym)
        # 2) hypernym is the ancestor of the hyponym
        elif lowest_common_ancestor == hypernym:
            dist = nx.shortest_path_length(graph, hypernym, hyponym)
        # 3) these two entities are the co-hyponym
        else:
            dist = - nx.shortest_path_length(graph, lowest_common_ancestor, hypernym)
        return dtype(dist)

    def hyponymy_distance_fast(self, hypernym, hyponym, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        ancestors_hypernym = nx.ancestors(self.dag, hypernym)
        ancestors_hyponym = nx.ancestors(self.dag, hyponym)
        ancestors_common = ancestors_hypernym.intersection(ancestors_hyponym)

        # 1) hypernym is the ancestor of the hyponym
        if hypernym in ancestors_hyponym:
            dist = nx.shortest_path_length(graph, hypernym, hyponym)
        # 2) hyponym is the ancestor of the hypernym (=reverse hyponymy)
        elif hyponym in ancestors_hypernym:
            dist = - nx.shortest_path_length(graph, hyponym, hypernym)
        # 3) not connected
        elif len(ancestors_common) == 0:
            dist = - self.depth(hypernym)
        # 4) these two entities are the co-hyponym
        elif len(ancestors_common) > 0:
            lst_path_length = (nx.shortest_path_length(graph, common, hypernym) for common in ancestors_common)
            dist = - min(lst_path_length)
        return dtype(dist)

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True,
                            candidate_weight_function = None) -> List[str]:
        graph = self.dag
        if entity not in graph:
            return []

        if exclude_hypernyms:
            non_candidates = self.ancestors_and_descendents(entity)
        else:
            non_candidates = self.descendents(entity)
        candidates = self._nodes if candidates is None else set(candidates).intersection(self._nodes)
        candidates = candidates - non_candidates

        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [next(iter(candidates))]*size

        # sampling with replacement
        # if `candidate_weight_function` is specified, then weighted sampling
        candidates = list(candidates)
        if candidate_weight_function is None:
            sampled = np.random.choice(candidates, size=size)
        else:
            vec_weights = np.fromiter(map(candidate_weight_function, candidates), dtype=np.float)
            vec_weights = vec_weights / np.sum(vec_weights)
            sampled = np.random.choice(candidates, size=size, p=vec_weights)

        sampled = sampled.tolist()

        return sampled

    def sample_random_hyponyms(self, entity: str,
                               candidates: Optional[Iterable[str]] = None,
                               size: int = 1, exclude_hypernyms: bool = True,
                               weighted_sampling: bool = False):
        if weighted_sampling:
            weight_function = lambda e: self.hyponym_frequency(e, not_exist=1)
        else:
            weight_function = None
        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms,
                                                             candidate_weight_function=weight_function)
        lst_ret = [(entity, hyponym, self.hyponymy_distance_fast(entity, hyponym)) for hyponym in lst_non_hyponymy_entities]

        return lst_ret

    def sample_random_hypernyms(self, entity: str,
                                candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True,
                                weighted_sampling: bool = False):

        if weighted_sampling:
            weight_function = lambda e: self.hypernym_frequency(e, not_exist=1)
        else:
            weight_function = None
        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms,
                                                             candidate_weight_function=weight_function)
        lst_ret = [(hypernym, entity, self.hyponymy_distance_fast(hypernym, entity)) for hypernym in lst_non_hyponymy_entities]

        return lst_ret

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None):
        graph = self.dag
        if (hypernym not in graph) or (hyponym not in graph):
            return not_exists

        if include_reverse_hyponymy:
            candidates = self.hyponyms(hypernym) | self.hypernyms(hypernym)
        else:
            candidates = self.hyponyms(hypernym)
        ret = hyponym in candidates

        return ret


class WordNetTaxonomy(BasicTaxonomy):

    def __init__(self, hyponymy_dataset: Optional[HyponymyDataset] = None):

        # build taxonomy as for each part-of-speech tags as DAG
        if hyponymy_dataset is not None:
            set_entity_types = set((record["pos"] for record in hyponymy_dataset))
            dict_iter_hyponymy_pairs = {}
            for entity_type in set_entity_types:
                generator = [(record["synset_hypernym"], record["synset_hyponym"]) for record in hyponymy_dataset if (record["distance"] == 1.0) and (record["pos"] == entity_type)]
                dict_iter_hyponymy_pairs[entity_type] = generator

            self.build_directed_acyclic_graph(dict_iter_hyponymy_pairs)
        else:
            self._dag = {}
            self._cache_root_nodes = {}
            self._active_entity_type = None
            warnings.warn("argument `hyponymy_dataset` was not specified. you must call `build_directed_acyclic_graph()` manually.")

    def _remove_redundant_edges(self, graph):

        n_removed = 0
        q = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        while True:
            try:
                loops = nx.find_cycle(graph)
                graph.remove_edge(*loops[-1])
                n_removed += 1
                q.update(n_removed)
            except:
                break

        print(f"number of removed edges: {n_removed}")
        return graph

    def build_directed_acyclic_graph(self, dict_iter_hyponymy_pairs: Dict[str, Iterable[Tuple[str, str]]]):
        self._dag = {}
        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            print(f"building graph. entity type: {entity_type}")
            graph = nx.DiGraph()
            graph.add_edges_from(iter_hyponymy_pairs)
            print(f"removing redundant edges to make it acyclic graph...")
            graph = self._remove_redundant_edges(graph)
            self._dag[entity_type] = graph

        self._active_entity_type = None
        self._cache_root_nodes = {}

    def hypernyms(self, entity, part_of_speech):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hypernyms(entity)

    def hyponyms(self, entity, part_of_speech):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hyponyms(entity)

    def depth(self, entity, part_of_speech, offset=1, not_exists=None):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().depth(entity, offset, not_exists)

    def hyponymy_distance(self, hypernym, hyponym, part_of_speech, dtype: Type = float, not_exists=None):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hyponymy_distance(hypernym, hyponym, dtype)

    def hyponymy_distance_fast(self, hypernym, hyponym, part_of_speech, dtype: Type = float, not_exists=None):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().hyponymy_distance_fast(hypernym, hyponym, dtype)

    def sample_non_hyponymy(self, entity, part_of_speech, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True) -> List[str]:
        self.activate_entity_type(entity_type=part_of_speech)
        return super().sample_non_hyponymy(entity, candidates, size, exclude_hypernyms)

    def sample_random_hyponyms(self, hypernym, part_of_speech, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True):
        self.activate_entity_type(entity_type=part_of_speech)
        return super().sample_random_hyponyms(hypernym, candidates, size, exclude_hypernyms)

    @property
    def active_entity_type(self):
        return self._active_entity_type

    def activate_entity_type(self, entity_type):
        self._active_entity_type = entity_type

    @property
    def dag(self):
        if self.active_entity_type is not None:
            return self._dag[self.active_entity_type]
        else:
            return self._dag

    @property
    def entity_types(self):
        return set(self._dag.keys())