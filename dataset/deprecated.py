#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Tuple, Dict, Optional, Union
import random

from .lexical_knowledge import HyponymyDataset
from .embeddings_plus_lexical_knowledge import WordEmbeddingsAndHyponymyDataset, AbstractWordEmbeddingsDataset

class WordEmbeddingsAndHyponymyDatasetWithNonHyponymyRelation(WordEmbeddingsAndHyponymyDataset):

    def __init__(self, word_embeddings_dataset: AbstractWordEmbeddingsDataset, hyponymy_dataset: HyponymyDataset,
                 embedding_batch_size: int, hyponymy_batch_size: int, non_hyponymy_batch_size: Optional[int] = None,
                 non_hyponymy_relation_distance: float = -1.0,
                 exclude_reverse_hyponymy_from_non_hyponymy_relation: bool = True,
                 verbose: bool = False, **kwargs_hyponymy_dataloader):

        super().__init__(word_embeddings_dataset, hyponymy_dataset,
                 embedding_batch_size, hyponymy_batch_size, verbose, **kwargs_hyponymy_dataloader)

        self._non_hyponymy_batch_size = hyponymy_batch_size if non_hyponymy_batch_size is None else non_hyponymy_batch_size
        self._non_hyponymy_relation_distance = non_hyponymy_relation_distance

        # build the set of hyponymy relation; it will be used to sample non-hyponymy relation
        self._build_hyponymy_relation_set(include_reverse_hyponymy_relation=exclude_reverse_hyponymy_from_non_hyponymy_relation)

    def _build_hyponymy_relation_set(self, include_reverse_hyponymy_relation: bool):

        set_hyponymy_relation = set()
        for batch in self._hyponymy_dataset:
            tup = (batch["hyponym"], batch["hypernym"])
            set_hyponymy_relation.add(tup)
            if include_reverse_hyponymy_relation:
                tup_rev = (batch["hypernym"], batch["hyponym"])
                set_hyponymy_relation.add(tup_rev)

        self._set_hyponymy_relation = set_hyponymy_relation

    def _create_non_hyponymy_relation_in_the_minibatch(self, lst_entities: List[str], size: int) -> List[Tuple[int, int, float]]:

        entity_to_index = {token:idx for idx, token in enumerate(lst_entities)}

        # create non-hyponymy relation
        lst_entity_tuple = []
        while True:
            entity_i, entity_j = random.sample(lst_entities, 2)

            if (entity_i, entity_j) in self._set_hyponymy_relation:
                continue

            idx_i = entity_to_index[entity_i]
            idx_j = entity_to_index[entity_j]
            dist_ij = self._non_hyponymy_relation_distance

            lst_entity_tuple.append((idx_i, idx_j, dist_ij))

            if len(lst_entity_tuple) >= size:
                break

        return lst_entity_tuple

    def __getitem__(self, idx):

        while True:
            n_idx_min = self._hyponymy_batch_size * idx
            n_idx_max = self._hyponymy_batch_size * (idx+1)

            # feed hyponymy relation from the hyponymy dataset
            batch_hyponymy_b = self._hyponymy_dataset[n_idx_min:n_idx_max]
            # remove hyponymy pairs which is not encodable
            batch_hyponymy = [sample for sample in batch_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]
            if len(batch_hyponymy) == 0:
                idx += 1
                continue

            # create a minibatch based on the valid hyponymy relation
            # in this process, entities which are not in the hyponymy dataset are randomly sampled
            batch = self._create_batch_from_hyponymy_samples(batch_hyponymy=batch_hyponymy)

            # we randomly sample the non-hyponymy relation from the mini-batch
            lst_entities = batch["entity"]

            batch["non_hyponymy_relation"] = self._create_non_hyponymy_relation_in_the_minibatch(lst_entities=lst_entities, size=self._non_hyponymy_batch_size)

            return batch

    def __iter__(self):

        for batch_hyponymy_b in self._hyponymy_dataloader:
            # remove hyponymy pairs which is not encodable
            batch_hyponymy = [sample for sample in batch_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]
            if len(batch_hyponymy) == 0:
                continue

            batch = self._create_batch_from_hyponymy_samples(batch_hyponymy=batch_hyponymy)
            lst_entities = batch["entity"]

            lst_non_hyponymy_relations = self._create_non_hyponymy_relation_in_the_minibatch(lst_entities=lst_entities)
            batch["non_hyponymy_relation"] = lst_non_hyponymy_relations

            yield batch
