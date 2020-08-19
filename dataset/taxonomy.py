#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union
from collections import defaultdict, Counter
from functools import lru_cache
import warnings
import networkx as nx
import numpy as np
import math
import random

from .lexical_knowledge import HyponymyDataset
from ._taxonomy_experiment import BasicHyponymyPairSet, WordNetHyponymyPairSet

class BasicTaxonomy(object):

    _DEBUG_MODE = False

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as a DAG
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset if record["distance"] == 1.0)
        self.build_directed_acyclic_graph(iter_hyponymy_pairs)
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset)
        self.record_ancestors_and_descendants(iter_hyponymy_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    @property
    def dag(self):
        return self._dag

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

    def build_directed_acyclic_graph(self, iter_hyponymy_pairs: Iterable[Tuple[str, str]]):
        """
        build taxonomy as a DAG based on the set of hyponymy relations

        @param iter_hyponymy_pairs: iterable of the tuple (hypernym, hyponym)
        """
        graph = nx.DiGraph()
        graph.add_edges_from(iter_hyponymy_pairs)

        self._dag = graph
        self._cache_root_nodes = {}
        self._nodes = tuple(graph.nodes)

    def record_ancestors_and_descendants(self, iter_hyponymy_pairs):
        self._trainset_ancestors = defaultdict(set)
        self._trainset_descendants = defaultdict(set)
        for hypernym, hyponym in iter_hyponymy_pairs:
            self._trainset_ancestors[hyponym].add(hypernym)
            self._trainset_descendants[hypernym].add(hyponym)

    def _random_number_generator_iterator(self, max_value: int):
        seeds = np.arange(max_value)
        while True:
            np.random.shuffle(seeds)
            for idx in seeds:
                yield idx

    def _find_root_nodes(self, graph) -> Set[str]:
        hash_value = graph.__hash__() + graph.number_of_nodes()
        if hash_value in self._cache_root_nodes:
            return self._cache_root_nodes[hash_value]

        root_nodes = set([k for k,v in graph.in_degree() if v == 0])
        self._cache_root_nodes[hash_value] = root_nodes
        return root_nodes

    def dag_ancestors(self, entity):
        return self._dag_ancestors(entity, self.dag)

    #@lru_cache(maxsize=1000000)
    def _dag_ancestors(self, entity, graph):
        return nx.ancestors(graph, entity)

    def hypernyms(self, entity):
        return self.dag_ancestors(entity).union(self.trainset_ancestors.get(entity, set()))

    def hyponyms(self, entity):
        return nx.descendants(self.dag, entity).union(self.trainset_descendants.get(entity, set()))

    def hypernyms_and_hyponyms_and_self(self, entity):
        return self._hypernyms_and_hyponyms_and_self(entity, self.dag)

    #@lru_cache(maxsize=1000000)
    def _hypernyms_and_hyponyms_and_self(self, entity, graph):
        return self.hyponyms(entity) | self.hypernyms(entity) | {entity}

    def hyponyms_and_self(self, entity):
        return self._hyponyms_and_self(entity, self.dag)

    #@lru_cache(maxsize=1000000)
    def _hyponyms_and_self(self, entity, graph):
        return self.hyponyms(entity) | {entity}

    def co_hyponyms(self, entity):
        graph = self.dag
        if entity not in graph:
            return {}
        direct_root_nodes = nx.ancestors(graph, entity) & self._find_root_nodes(graph)
        branches = map(lambda entity: nx.descendants(self.dag, entity), direct_root_nodes)
        branches = set().union(*branches)
        co_hyponyms = branches - self.hypernyms_and_hyponyms_and_self(entity)
        return co_hyponyms

    def depth(self, entity, offset=1, not_exists=None, **kwargs):
        return self._depth(entity, self.dag, offset, not_exists)

    #@lru_cache(maxsize=1000000)
    def _depth(self, entity, graph, offset=1, not_exists=None):
        if entity not in graph:
            return not_exists

        direct_root_nodes = self.dag_ancestors(entity) & self._find_root_nodes(graph)
        if len(direct_root_nodes) == 0:
            depth = 0
        else:
            f_path_length_to_entity = lambda source: nx.shortest_path_length(graph, source, entity)
            depth = max(map(f_path_length_to_entity, direct_root_nodes))

        return depth + offset

    def hyponymy_score_slow(self, hypernym, hyponym, dtype: Type = float):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        lowest_common_ancestor = nx.lowest_common_ancestor(graph, hypernym, hyponym)

        # 1) hypernym is the ancestor of the hyponym (=hyponymy)
        if nx.has_path(graph, hypernym, hyponym):
            dist = nx.shortest_path_length(graph, hypernym, hyponym)
        # 2) hyponym is the ancestor of the hypernym (=reverse hyponymy)
        elif nx.has_path(graph, hyponym, hypernym):
            dist = - nx.shortest_path_length(graph, hyponym, hypernym)
        # 3) these two entities are the co-hyponym
        elif lowest_common_ancestor is not None:
            dist = - nx.shortest_path_length(graph, lowest_common_ancestor, hypernym)
        # 4) other
        else:
            dist = - self.depth(hypernym)

        return dtype(dist)

    def hyponymy_score(self, hypernym, hyponym, dtype: Type = float, **kwargs):
        graph = self.dag
        if (hypernym not in graph) or (hyponym not in graph):
            if self._DEBUG_MODE:
                raise ValueError(f"invalid node is specified: {hypernym}, {hyponym}")
            else:
                return None

        ancestors_hypernym = self.dag_ancestors(hypernym)
        ancestors_hyponym = self.dag_ancestors(hyponym)
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
            depth_lca = max(map(self.depth, ancestors_common))
            dist = depth_lca - self.depth(hypernym)
        return dtype(dist)

    def lowest_common_ancestor_depth(self, hypernym, hyponym, offset: int =1, dtype: Type = float, **kwargs):
        graph = self.dag
        if hypernym not in graph:
            raise ValueError(f"invalid node is specified: {hypernym}")
        if hyponym not in graph:
            raise ValueError(f"invalid node is specified: {hyponym}")

        ancestors_hypernym = self.dag_ancestors(hypernym)
        ancestors_hyponym = self.dag_ancestors(hyponym)
        ancestors_common = ancestors_hypernym.intersection(ancestors_hyponym)

        # 1) hypernym is the ancestor of the hyponym: LCA is hypernym
        if hypernym in ancestors_hyponym:
            depth_lca = self.depth(entity=hypernym, offset=offset)
        # 2) hyponym is the ancestor of the hypernym (=reverse hyponymy): LCA is hyponym
        elif hyponym in ancestors_hypernym:
            depth_lca = self.depth(entity=hyponym, offset=offset)
        # 3) not connected -> LCA is empty
        elif len(ancestors_common) == 0:
            depth_lca = 0
        # 4) these two entities are the co-hyponym: LCA is the deepest co-hyponym.
        elif len(ancestors_common) > 0:
            lst_depth = (self.depth(entity=common, offset=offset) for common in ancestors_common)
            depth_lca = max(lst_depth)
        return dtype(depth_lca)

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True) -> List[str]:
        graph = self.dag
        if entity not in graph:
            return []

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
        lst_ret = [(entity, hyponym, self.hyponymy_score(entity, hyponym)) for hyponym in lst_non_hyponymy_entities]

        return lst_ret

    def sample_random_hypernyms(self, entity: str,
                                candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True, **kwargs):

        lst_non_hyponymy_entities = self.sample_non_hyponymy(entity=entity, candidates=candidates,
                                                             size=size, exclude_hypernyms=exclude_hypernyms)
        lst_ret = [(hypernym, entity, self.hyponymy_score(hypernym, entity)) for hypernym in lst_non_hyponymy_entities]

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

    def sample_random_co_hyponyms(self, hypernym: str, hyponym: str, size: int = 1, break_probability: float = 0.8, **kwargs):
        graph = self.dag
        lst_co_hyponymy = []
        if (hypernym not in graph) or (hyponym not in graph):
            return lst_co_hyponymy
        if not nx.has_path(graph, source=hypernym, target=hyponym):
            return lst_co_hyponymy

        for _ in range(size):
            co_hyponymy_triple = self._sample_random_co_hyponymy(hypernym, hyponym, break_probability)
            if co_hyponymy_triple is not None:
                lst_co_hyponymy.append(co_hyponymy_triple)
        return lst_co_hyponymy

    def _sample_random_co_hyponymy(self, hypernym: str, hyponym: str, break_probability: float) -> Tuple[str, str, float]:
        graph = self.dag
        shortest_path = self.hypernyms(hyponym) - self.hypernyms(hypernym)
        non_candidates = self.hyponyms_and_self(hyponym)
        children = set(graph.successors(hypernym)) - non_candidates
        hyponymy_score = None

        while len(children) > 0:
            node = random.sample(children, 1)[0]
            if node not in shortest_path:
                hyponymy_score = -1 if hyponymy_score is None else hyponymy_score - 1
                q = random.uniform(0,1)
                if q <= break_probability:
                    break

            # update children
            children = set(graph.successors(node)) - non_candidates

        if hyponymy_score is None:
            return None
        else:
            hyponymy_score = self.hyponymy_score(hypernym=node, hyponym=hyponym)
            return (node, hyponym, hyponymy_score)


class WordNetTaxonomy(BasicTaxonomy):

    def __init__(self, hyponymy_dataset: Optional[HyponymyDataset] = None):

        # build taxonomy as for each part-of-speech tags as DAG
        dict_iter_hyponymy_pairs = defaultdict(list)
        dict_iter_trainset_pairs = defaultdict(list)
        for record in hyponymy_dataset:
            entity_type = record["pos"]
            entity_hyper = record["hypernym"]
            entity_hypo = record["hyponym"]

            dict_iter_trainset_pairs[entity_type].append((entity_hyper, entity_hypo))
            if record["distance"] == 1.0:
                dict_iter_hyponymy_pairs[entity_type].append((entity_hyper, entity_hypo))

        self.build_directed_acyclic_graph(dict_iter_hyponymy_pairs)
        self.record_ancestors_and_descendants(dict_iter_trainset_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    def build_directed_acyclic_graph(self, dict_iter_hyponymy_pairs: Dict[str, Iterable[Tuple[str, str]]]):
        self._dag = {}
        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            print(f"building graph. entity type: {entity_type}")
            graph = nx.DiGraph()
            graph.add_edges_from(iter_hyponymy_pairs)
            self._dag[entity_type] = graph
            # assert nx.is_directed_acyclic_graph(graph), f"failed to construct directed acyclic graph."

        self._active_entity_type = None
        self._cache_root_nodes = {}
        self._nodes = {entity_type:tuple(graph.nodes) for entity_type, graph in self._dag.items()}

    def record_ancestors_and_descendants(self, dict_iter_hyponymy_pairs):
        self._trainset_ancestors = defaultdict(lambda :defaultdict(set))
        self._trainset_descendants = defaultdict(lambda :defaultdict(set))
        for entity_type, iter_hyponymy_pairs in dict_iter_hyponymy_pairs.items():
            for hypernym, hyponym in iter_hyponymy_pairs:
                self._trainset_ancestors[entity_type][hyponym].add(hypernym)
                self._trainset_descendants[entity_type][hypernym].add(hyponym)

    def depth(self, entity, offset=1, not_exists=None, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().depth(entity, offset, not_exists)

    def lowest_common_ancestor_depth(self, hypernym, hyponym, offset: int =1, dtype: Type = float, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().lowest_common_ancestor_depth(hypernym, hyponym, offset, dtype)

    def hyponymy_score_slow(self, hypernym, hyponym, dtype: Type = float, not_exists=None, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        if not nx.is_directed_acyclic_graph(self.dag):
            raise NotImplementedError(f"you can't use this method.")
        return super().hyponymy_score_slow(hypernym, hyponym, dtype)

    def hyponymy_score(self, hypernym, hyponym, dtype: Type = float, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().hyponymy_score(hypernym, hyponym, dtype)

    def sample_random_hypernyms(self, entity: str, candidates: Optional[Iterable[str]] = None,
                                size: int = 1, exclude_hypernyms: bool = True, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().sample_random_hypernyms(entity, candidates, size, exclude_hypernyms)

    def sample_random_hyponyms(self, entity: str, candidates: Optional[Iterable[str]] = None,
                               size: int = 1, exclude_hypernyms: bool = True, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().sample_random_hyponyms(entity, candidates, size, exclude_hypernyms)

    def sample_random_co_hyponyms(self, hypernym: str, hyponym: str, size: int = 1, break_probability: float = 0.5, **kwargs):
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().sample_random_co_hyponyms(hypernym, hyponym, size, break_probability)

    @property
    def ACTIVE_ENTITY_TYPE(self):
        return self._active_entity_type

    @ACTIVE_ENTITY_TYPE.setter
    def ACTIVE_ENTITY_TYPE(self, value):
        if value is not None:
            self._active_entity_type = value

    @property
    def dag(self):
        return self._dag.get(self.ACTIVE_ENTITY_TYPE, self._dag)

    @property
    def entity_types(self):
        return set(self._dag.keys())

    @property
    def trainset_ancestors(self):
        return self._trainset_ancestors.get(self.ACTIVE_ENTITY_TYPE, self._trainset_ancestors)

    @property
    def trainset_descendants(self):
        return self._trainset_descendants.get(self.ACTIVE_ENTITY_TYPE, self._trainset_descendants)

    @property
    def nodes(self):
        return self._nodes.get(self.ACTIVE_ENTITY_TYPE, self._nodes)

    @property
    def n_nodes_max(self):
        return max(map(len, self._nodes.values()))


class SynsetAwareWordnetTaxonomy(WordNetTaxonomy):

    _SEPARATOR = "▁" # U+2581

    def __init__(self, hyponymy_dataset: Optional[HyponymyDataset] = None):

        # build taxonomy as for each part-of-speech tags as DAG
        dict_iter_hyponymy_pairs = defaultdict(list)
        dict_iter_trainset_pairs = defaultdict(list)
        for record in hyponymy_dataset:
            entity_type = record["pos"]
            lemma_hyper = record["hypernym"]
            lemma_hypo = record["hyponym"]
            synset_hyper = record["synset_hypernym"]
            synset_hypo = record["synset_hyponym"]
            entity_hyper = self.synset_and_lemma_to_entity(synset_hyper, lemma_hyper)
            entity_hypo = self.synset_and_lemma_to_entity(synset_hypo, lemma_hypo)

            dict_iter_trainset_pairs[entity_type].append((entity_hyper, entity_hypo))
            if record["distance"] == 1.0:
                dict_iter_hyponymy_pairs[entity_type].append((entity_hyper, entity_hypo))

        self.build_directed_acyclic_graph(dict_iter_hyponymy_pairs)
        self.record_ancestors_and_descendants(dict_iter_trainset_pairs)

    def synset_and_lemma_to_entity(self, synset: str, lemma: str):
        return synset + self._SEPARATOR + lemma

    def entity_to_lemma(self, entity: str):
        return entity[entity.find(self._SEPARATOR)+1:]

    def entity_to_synset_and_lemma(self, entity: str):
        return entity.split(self._SEPARATOR)

    def search_entities_by_lemma(self, lemma: str, part_of_speech: str):
        self.ACTIVE_ENTITY_TYPE = part_of_speech
        key = self._SEPARATOR + lemma
        return {entity for entity in self.nodes if entity.endswith(key)}