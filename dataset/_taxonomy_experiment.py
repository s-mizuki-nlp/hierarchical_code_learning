#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys, io, os
from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union
from collections import defaultdict
import numpy as np
import math
import progressbar
import hashlib

import pickle

from .lexical_knowledge import HyponymyDataset
from .word_embeddings import AbstractWordEmbeddingsDataset
from .utils import EmbeddingSimilaritySearch

class BasicHyponymyPairSet(object):

    _DEBUG_MODE = False

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as a DAG
        iter_hyponymy_pairs = ((record["hypernym"], record["hyponym"], record["distance"]) for record in hyponymy_dataset)
        self.record_ancestors_and_descendants(iter_hyponymy_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    @property
    def nodes(self):
        return self._nodes

    @property
    def set_nodes(self):
        return self._set_nodes

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
    def hyponymies(self):
        return self._hyponymies

    @property
    def trainset_negative_nearest_neighbors(self):
        return self._trainset_negative_nearest_neighbors

    def record_ancestors_and_descendants(self, iter_hyponymy_pairs):
        self._hyponymies  = set() # set of the tuple of (hypernym, hyponym) pair.
        nodes = set()
        self._trainset_hyponymies_and_self = defaultdict(set)
        for hypernym, hyponym, distance in iter_hyponymy_pairs:
            nodes.add(hypernym)
            nodes.add(hyponym)

            if distance >= 0.0:
                self._trainset_hyponymies_and_self[hyponym].add(hypernym)
                self._trainset_hyponymies_and_self[hypernym].add(hyponym)

            if distance >= 1.0:
                self._hyponymies.add((hypernym, hyponym))

        # insert oneself
        for entity in self._trainset_hyponymies_and_self.keys():
            self._trainset_hyponymies_and_self[entity].add(entity)

        # set nodes
        self._set_nodes = nodes
        self._nodes = tuple(nodes)

    def prebuild_negative_nearest_neighbors(self, word_embeddings_dataset: AbstractWordEmbeddingsDataset, top_k: Optional[int] = None, top_q: Optional[float] = None):
        print(f"lookup negative nearest neighbors. entity size:{len(self.nodes)}")
        self._trainset_negative_nearest_neighbors = {}

        # get word embeddings of the nodes
        iter_embeddings = (word_embeddings_dataset[entity] for entity in self.nodes)
        embeddings = {obj["entity"]:obj["embedding"] for obj in iter_embeddings}

        # instanciate similarity search class
        entity_similarity_model = EmbeddingSimilaritySearch(embeddings=embeddings)

        # loop over whole nodes
        for entity in self.nodes:
            positive_entities = self.hypernyms_and_hyponyms_and_self(entity)
            vec_e = word_embeddings_dataset[entity]["embedding"]
            lst_similar_entities = entity_similarity_model.most_similar(vector=vec_e, top_k=top_k, top_q=top_q, excludes=positive_entities)
            self._trainset_negative_nearest_neighbors[entity] = tuple(e for e, sim in lst_similar_entities)

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

    def hypernyms_and_hyponyms_and_self(self, entity) -> Set[str]:
        return self.trainset_hyponymies_and_self.get(entity, set())

    def negative_nearest_neighbors(self, entity, **kwargs) -> Iterable[str]:
        return self.trainset_negative_nearest_neighbors.get(entity, tuple())

    def sample_non_hyponymy(self, entity, candidates: Optional[Iterable[str]] = None,
                            size: int = 1, exclude_hypernyms: bool = True) -> List[str]:

        if exclude_hypernyms:
            non_candidates = self.hypernyms_and_hyponyms_and_self(entity)
        else:
            non_candidates = self.hyponyms_and_self(entity)

        if candidates is None:
            # use whole entities as the candidates.
            candidates = self.nodes
            # remove non-candidates beforehand to make the negative sampling more efficient.
            non_candidate_ratio = len(non_candidates)/len(candidates)
            if 0.999 <= non_candidate_ratio:
                # give up sampling
                return []
            elif 0.9 <= non_candidate_ratio < 0.999:
                candidates = tuple(set(candidates) - non_candidates)
        else:
            # use user-specified candidates as it is.
            if len(candidates) == 0:
                return []
            else:
                pass

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
        return entity in self.set_nodes

    def is_hyponymy_relation(self, hypernym, hyponym, include_reverse_hyponymy: bool = True, not_exists = None, **kwargs):
        if not (self.is_in_dataset(hypernym) and self.is_in_dataset(hyponym)):
            return not_exists

        pair_forward = (hypernym, hyponym)
        pair_reverse = (hyponym, hypernym)

        if include_reverse_hyponymy:
            ret = (pair_forward in self.hyponymies) or (pair_reverse in self.hyponymies)
        else:
            ret = (pair_forward in self.hyponymies)

        return ret


class WordNetHyponymyPairSet(BasicHyponymyPairSet):

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        # build taxonomy as for each part-of-speech tags as DAG
        dict_iter_trainset_pairs = defaultdict(list)
        for record in hyponymy_dataset:
            entity_type = record["pos"]
            entity_hyper = record["hypernym"]
            entity_hypo = record["hyponym"]
            distance = record["distance"]
            dict_iter_trainset_pairs[entity_type].append((entity_hyper, entity_hypo, distance))

        self.record_ancestors_and_descendants(dict_iter_trainset_pairs)
        self._random_number_generator = self._random_number_generator_iterator(max_value=self.n_nodes_max)

    def __hash__(self):
        # compute persistent hash using nodes in the taxonomy.
        m = hashlib.md5()
        for entity_type in sorted(self.entity_types):
            m.update(entity_type.encode())
            for node in sorted(self._nodes[entity_type]):
                m.update(node.encode())

        return int(m.hexdigest(), 16)

    def record_ancestors_and_descendants(self, dict_iter_hyponymy_pairs):
        self._hyponymies = defaultdict(lambda :set())
        self._trainset_hyponymies_and_self = defaultdict(lambda :defaultdict(set))
        nodes = defaultdict(lambda : set())

        for entity_type, iter_hyponymy_tuples in dict_iter_hyponymy_pairs.items():
            for hypernym, hyponym, distance in iter_hyponymy_tuples:
                nodes[entity_type].add(hypernym)
                nodes[entity_type].add(hyponym)

                if distance >= 0.0:
                    self._trainset_hyponymies_and_self[entity_type][hyponym].add(hypernym)
                    self._trainset_hyponymies_and_self[entity_type][hypernym].add(hyponym)

                if distance >= 1.0:
                    self._hyponymies[entity_type].add((hypernym, hyponym))

            # add oneself
            for entity in self._trainset_hyponymies_and_self[entity_type].keys():
                self._trainset_hyponymies_and_self[entity_type][entity].add(entity)

        # create tuple of entities as nodes
        self._set_nodes = nodes
        self._nodes = {entity_type:tuple(entities) for entity_type, entities in nodes.items()}

        # unset active entity type
        self._active_entity_type = None

    def prebuild_negative_nearest_neighbors(self, word_embeddings_dataset: AbstractWordEmbeddingsDataset,
                                            top_k: Optional[int] = None, top_q: Optional[float] = None,
                                            use_cache: bool = True, cache_dir: str = "./_cache/",
                                            force_cache_file_name: str = None):
        print(f"lookup negative nearest neighbors...")

        # intialize internal attribute
        self._trainset_negative_nearest_neighbors = {}
        for entity_type in self.entity_types:
            self._trainset_negative_nearest_neighbors[entity_type] = {}

        if use_cache:
            if isinstance(force_cache_file_name, str):
                cache_file_name = force_cache_file_name
            else:
                cache_file_name = "_".join(map(str, [hash(self), hash(word_embeddings_dataset), top_k, top_q]))
            path = os.path.join(cache_dir, cache_file_name)
            if os.path.exists(path):
                print(f"load from the cache file:{path}")
                with io.open(path, mode="rb") as ifs:
                    self._trainset_negative_nearest_neighbors = pickle.load(ifs)
                return True
            else:
                print(f"cache will be saved as:{path}")

        for entity_type in self.entity_types:
            # switch entity type
            self.ACTIVE_ENTITY_TYPE = entity_type
            top_n = top_k if top_k is not None else math.ceil(len(self.nodes)*top_q)
            print(f"entity type:{entity_type}, entity size:{len(self.nodes)}, nearest neighbors per entity: {top_n}")

            # instanciate similarity search class
            embeddings = word_embeddings_dataset.entities_to_embeddings(self.nodes, ignore_encode_error=True)
            entity_similarity_model = EmbeddingSimilaritySearch(embeddings=embeddings)

            # find negative nearest neighbors
            q = progressbar.ProgressBar(max_value=len(self.nodes))
            for idx, entity in enumerate(self.nodes):
                q.update(idx+1)
                vec_e = word_embeddings_dataset.encode(entity)
                if vec_e is None:
                    continue
                positive_entities = self.hypernyms_and_hyponyms_and_self(entity)
                lst_similar_entities = entity_similarity_model.most_similar(vector=vec_e, top_k=top_n, excludes=positive_entities)
                self._trainset_negative_nearest_neighbors[entity_type][entity] = tuple(e for e, sim in lst_similar_entities)

        # unset active entity type
        self._active_entity_type = None

        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with io.open(path, mode="wb") as ofs:
                pickle.dump(self._trainset_negative_nearest_neighbors, ofs)

    @property
    def ACTIVE_ENTITY_TYPE(self):
        return self._active_entity_type

    @ACTIVE_ENTITY_TYPE.setter
    def ACTIVE_ENTITY_TYPE(self, value):
        if value is not None:
            self._active_entity_type = value

    @property
    def entity_types(self):
        return set(self._hyponymies.keys())

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
    def set_nodes(self):
        return self._set_nodes.get(self.ACTIVE_ENTITY_TYPE, self._set_nodes)

    @property
    def trainset_hyponymies_and_self(self):
        return self._trainset_hyponymies_and_self.get(self.ACTIVE_ENTITY_TYPE, self._trainset_hyponymies_and_self)

    @property
    def trainset_negative_nearest_neighbors(self):
        return self._trainset_negative_nearest_neighbors.get(self.ACTIVE_ENTITY_TYPE, self._trainset_negative_nearest_neighbors)

    @property
    def hyponymies(self):
        return self._hyponymies.get(self.ACTIVE_ENTITY_TYPE, self._hyponymies)

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

    def negative_nearest_neighbors(self, entity, **kwargs) -> Iterable[str]:
        self.ACTIVE_ENTITY_TYPE = kwargs.get("part_of_speech", None)
        return super().negative_nearest_neighbors(entity)