#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional, Iterable, Tuple, Set, Type, List
import warnings
import networkx as nx
import random

from .lexical_knowledge import HyponymyDataset

class Taxonomy(object):

    def __init__(self, hyponymy_dataset: Optional[HyponymyDataset] = None):

        # build taxonomy as a DAG
        if hyponymy_dataset is not None:
            iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"]) for record in hyponymy_dataset if record["distance"] == 1.0)
            self.build_directed_acyclic_graph(iter_hyponymy_pairs)
        else:
            self._dag = None
            self._root_nodes = None
            warnings.warn("argument `hyponymy_dataset` was not specified. you must call `build_directed_acyclic_graph()` manually.")

    @property
    def dag(self):
        return self._dag

    def build_directed_acyclic_graph(self, iter_hyponymy_pairs: Iterable[Tuple[str, str]]):
        """
        build taxonomy as a DAG based on the set of hyponymy relations

        @param iter_hyponymy_pairs: iterable of the tuple (hypernym, hyponym)
        """
        graph = nx.DiGraph()
        graph.add_edges_from(iter_hyponymy_pairs)

        self._dag = graph
        self._root_nodes = self._find_root_nodes(graph=graph)

    def _find_root_nodes(self, graph) -> Set[str]:

        dict_in_degrees = graph.in_degree()
        root_nodes = set([k for k,v in dict_in_degrees.items() if v == 0])
        return root_nodes

    def hypernyms(self, entity):
        return nx.ancestors(self._dag, entity)

    def hyponyms(self, entity):
        return nx.descendants(self._dag, entity)

    def depth(self, entity, offset=1):
        graph = self._dag

        direct_root_nodes = nx.ancestors(graph, entity) & self._root_nodes
        if len(direct_root_nodes) == 0:
            depth = 0
        else:
            f_path_length_to_entity = lambda source: nx.shortest_path_length(graph, source, entity)
            depth = max(map(f_path_length_to_entity, direct_root_nodes))

        return depth + offset

    def hyponymy_distance(self, hypernym, hyponym, dtype: Type = float):
        graph = self._dag
        lowest_common_ancestor = nx.lowest_common_ancestor(graph, hypernym, hyponym)
        # 1) not connected
        if lowest_common_ancestor is None:
            dist = - self.depth(graph, hypernym)
        # 2) hypernym is the ancestor of the hyponym
        elif lowest_common_ancestor == hypernym:
            dist = - nx.shortest_path_length(graph, hypernym, hyponym)
        # 3) these two entities are the co-hyponym
        else:
            dist = - nx.shortest_path_length(graph, lowest_common_ancestor, hypernym)
        return dtype(dist)

    def sample_non_hyponym(self, entity, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True) -> List[str]:
        graph = self._dag
        if exclude_hypernyms:
            non_candidates = self.hyponyms(entity) | self.hypernyms(entity)
        else:
            non_candidates = self.hyponyms(entity)
        candidates = set(graph.nodes) if candidates is None else set(candidates)
        candidates = candidates - non_candidates

        if len(candidates) == 0:
            return []

        if len(candidates) < size:
            # sample with replacement
            sampled = random.choices(list(candidates), k=size)
        else:
            # sample without replacement
            sampled = random.sample(list(candidates), k=size)

        return sampled

    def sample_non_hyponymy_relations(self, hypernym, candidates: Optional[Iterable[str]] = None, size: int = 1, exclude_hypernyms: bool = True):

        lst_non_hyponyms = self.sample_non_hyponym(hypernym, candidates, size, exclude_hypernyms)
        lst_ret = [(hypernym, hyponym, self.hyponymy_distance(hypernym, hyponym)) for hyponym in lst_non_hyponyms]

        return lst_ret

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True):
        if include_reverse_hyponymy:
            candidates = self.hyponyms(hypernym) | self.hypernyms(hypernym)
        else:
            candidates = self.hyponyms(hypernym)
        ret = hyponym in candidates

        return ret