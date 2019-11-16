#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io, os
from typing import Union, Collection, Optional
from pprint import pprint

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

from .word_embeddings import AbstractWordEmbeddingsDataset
from .lexical_knowledge import HyponymyDataset



class WordEmbeddingsAndHyponymyDataset(IterableDataset):

    def __init__(self, word_embeddings_dataset: AbstractWordEmbeddingsDataset, hyponymy_dataset: HyponymyDataset,
                 embedding_batch_size: int, hyponymy_batch_size: int, verbose: bool = False, **kwargs_hyponymy_dataloader):
        assert embedding_batch_size >= 2*hyponymy_batch_size, f"`embedding_batch_size` must be two times larger than `hyponymy_batch_size`."

        self._word_embeddings_dataset = word_embeddings_dataset
        self._hyponymy_dataset = hyponymy_dataset
        self._embedding_batch_size = embedding_batch_size
        self._hyponymy_batch_size = hyponymy_batch_size
        self._hyponymy_dataloader = DataLoader(hyponymy_dataset, collate_fn=lambda v:v, batch_size=hyponymy_batch_size, **kwargs_hyponymy_dataloader)
        self._verbose = verbose

        if verbose:
            self.verify_batch_sizes()

    def verify_batch_sizes(self):

        n_embeddings = len(self._word_embeddings_dataset)
        n_hyponymy = len(self._hyponymy_dataset)

        coef_effective_samples = 1 - 1/np.e
        n_iteration = n_hyponymy / self._hyponymy_batch_size
        n_embeddings_sample_in_batch = self._embedding_batch_size - 2*self._hyponymy_batch_size

        n_embeddings_used_in_epoch = coef_effective_samples * n_iteration * n_embeddings_sample_in_batch
        ratio = n_embeddings_used_in_epoch / n_embeddings

        balanced_embedding_batch_size = np.ceil((n_embeddings/(coef_effective_samples*n_iteration)) + 2*self._hyponymy_batch_size)

        print(f"embeddings: {n_embeddings}")
        print(f"embeddings used in epoch: {n_embeddings_used_in_epoch:.0f}")
        print(f"consumption ratio: {ratio:2.3f}")
        print(f"balanced `embedding_batch_size` value: {balanced_embedding_batch_size:.0f}")

    def is_encodable_all(self, *tokens):
        return all([self._word_embeddings_dataset.is_encodable(token) for token in tokens])

    def __iter__(self):

        n_embeddings = len(self._word_embeddings_dataset)
        for batch_hyponymy_b in self._hyponymy_dataloader:
            # remove hyponymy pairs which is not encodable
            batch_hyponymy = [sample for sample in batch_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]
            if len(batch_hyponymy) == 0:
                continue

            # take tokens from hyponymy pairs
            set_tokens = set()
            for token_field in ("hyponym", "hypernym"):
                set_tokens.update(sample[token_field] for sample in batch_hyponymy)

            # take remaining tokens randomly from word embeddings dataset vocabulary
            n_diff = self._embedding_batch_size - len(set_tokens)
            lst_index = np.random.choice(range(n_embeddings), size=n_diff, replace=False)
            lst_tokens_from_embeddings = [self._word_embeddings_dataset.index_to_entity(idx) for idx in lst_index]
            # add to token set
            set_tokens.update(lst_tokens_from_embeddings)

            # create temporary token-to-index mapping
            lst_tokens = list(set_tokens)
            token_to_index = {token:idx for idx, token in enumerate(lst_tokens)}

            # create embeddings
            mat_embeddings = np.stack([self._word_embeddings_dataset[token]["embedding"] for token in lst_tokens])

            # create hyponymy relations
            lst_hyponymy_relation = []
            for hyponymy in batch_hyponymy:
                idx_hypo = token_to_index[hyponymy["hyponym"]]
                idx_hyper = token_to_index[hyponymy["hypernym"]]
                distance = hyponymy["distance"]
                lst_hyponymy_relation.append((idx_hypo, idx_hyper, distance))

            batch = {
                "embedding": mat_embeddings,
                "entity": lst_tokens,
                "hyponymy_relation": lst_hyponymy_relation
            }
            if self._verbose:
                batch["entity_sampled"] = lst_tokens_from_embeddings
                batch["hyponymy_relation_raw"] = batch_hyponymy_b

            yield batch