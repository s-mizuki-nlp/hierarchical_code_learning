#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Dict, Optional, Union
import random

from torch.utils.data import Dataset, DataLoader
import numpy as np

from .word_embeddings import AbstractWordEmbeddingsDataset
from .lexical_knowledge import HyponymyDataset, WordNetHyponymyDataset
from .taxonomy import BasicHyponymyPairSet
from .embeddings_plus_lexical_knowledge import WordEmbeddingsAndHyponymyDataset


class WordEmbeddingsAndHyponymyDatasetWithNonHyponymyRelation(WordEmbeddingsAndHyponymyDataset):

    def __init__(self, word_embeddings_dataset: AbstractWordEmbeddingsDataset, hyponymy_dataset: HyponymyDataset,
                 embedding_batch_size: int, hyponymy_batch_size: int, non_hyponymy_batch_size: Optional[int] = None,
                 non_hyponymy_relation_target: Union[str, List[str]] = ["hyponym","hypernym"],
                 exclude_reverse_hyponymy_from_non_hyponymy_relation: bool = True,
                 swap_hyponymy_relations: bool = False,
                 limit_hyponym_candidates_within_minibatch: bool = False,
                 verbose: bool = False, **kwargs_hyponymy_dataloader):

        super().__init__(word_embeddings_dataset, hyponymy_dataset,
                         embedding_batch_size, hyponymy_batch_size,
                         entity_depth_information = None,
                         verbose=False, **kwargs_hyponymy_dataloader)

        # overwrite the taxonomy that was built by superclass
        if isinstance(hyponymy_dataset, WordNetHyponymyDataset):
            raise NotImplementedError(f"not implemented it yet.")
        elif isinstance(hyponymy_dataset, HyponymyDataset):
            self._taxonomy = BasicHyponymyPairSet(hyponymy_dataset=hyponymy_dataset)
        else:
            raise NotImplementedError(f"unsupported hyponymy dataset type: {type(hyponymy_dataset)}")

        assert non_hyponymy_batch_size % hyponymy_batch_size == 0, f"`non_hyponymy_batch_size` must be a multiple of `hyponymy_batch_size`."
        if not limit_hyponym_candidates_within_minibatch:
            assert embedding_batch_size >= 2*hyponymy_batch_size + non_hyponymy_batch_size, \
            f"`embedding_batch_size` must be larger than `2*hyponymy_batch_size + non_hyponymy_batch_size`."

        available_options = ("hyponym","hypernym")
        if isinstance(non_hyponymy_relation_target, str):
            non_hyponymy_relation_target = non_hyponymy_relation_target.split(",")
        for target in non_hyponymy_relation_target:
            assert target in available_options, f"`non_hyponymy_relation_target` accepts: {','.join(available_options)}"

        self._non_hyponymy_batch_size = hyponymy_batch_size if non_hyponymy_batch_size is None else non_hyponymy_batch_size
        self._non_hyponymy_multiple = non_hyponymy_batch_size // hyponymy_batch_size
        self._non_hyponymy_relation_target = non_hyponymy_relation_target
        self._exclude_reverse_hyponymy_from_non_hyponymy_relation = exclude_reverse_hyponymy_from_non_hyponymy_relation
        self._limit_hyponym_candidates_within_minibatch = limit_hyponym_candidates_within_minibatch
        self._swap_hyponymy_relations = swap_hyponymy_relations
        self._verbose = verbose

        if len(non_hyponymy_relation_target) > 1:
            n_mod = len(non_hyponymy_relation_target)
            assert self._non_hyponymy_multiple % n_mod == 0, \
                f"When you specify multiple `non_hyponymy_relation_target`, `non_hyponymy_batch_size` must be a multiple of the `hyponymy_batch_size`"

        if verbose:
            self.verify_batch_sizes()

    def verify_batch_sizes(self):

        n_embeddings = len(self._word_embeddings_dataset)
        n_hyponymy = len(self._hyponymy_dataset)

        coef_effective_samples = 1 - 1/np.e
        n_iteration = n_hyponymy / self._hyponymy_batch_size
        if self._limit_hyponym_candidates_within_minibatch:
            n_embeddings_sample_in_batch = self._embedding_batch_size - 2*self._hyponymy_batch_size
            multiplier = 2
        else:
            n_embeddings_sample_in_batch = self._embedding_batch_size - 2*self._hyponymy_batch_size - self._non_hyponymy_batch_size
            multiplier = 2 + self._non_hyponymy_multiple

        n_embeddings_used_in_epoch = coef_effective_samples * n_iteration * n_embeddings_sample_in_batch
        ratio = n_embeddings_used_in_epoch / n_embeddings

        balanced_embeddings_sample_in_batch = (n_embeddings/coef_effective_samples) / n_iteration
        balanced_embedding_batch_size = np.ceil(self._embedding_batch_size - (n_embeddings_sample_in_batch - balanced_embeddings_sample_in_batch))
        balanced_hyponymy_batch_size = np.floor(n_hyponymy*self._embedding_batch_size*coef_effective_samples/(n_embeddings+multiplier*coef_effective_samples*n_hyponymy))

        print(f"hyponymy relations: {n_hyponymy}")
        print(f"non-hyponymy relations: {n_hyponymy*self._non_hyponymy_multiple}")
        print(f"embeddings: {n_embeddings}")
        print(f"embeddings referenced in epoch: {n_embeddings_used_in_epoch:.0f}")
        print(f"consumption ratio: {ratio:2.3f}")
        print(f"balanced `embedding_batch_size` value: {balanced_embedding_batch_size:.0f}")
        print(f"balanced `hyponymy_batch_size` value: {balanced_hyponymy_batch_size:.0f}")
        print(f"balanced `non_hyponymy_batch_size value: {balanced_hyponymy_batch_size*self._non_hyponymy_multiple:.0f}")

    def _create_non_hyponymy_samples_from_hyponymy_samples(self, batch_hyponymy: List[Dict[str,Union[str,float]]],
                                                            size_per_sample: int) -> List[Dict[str,Union[str,float]]]:

        # (optional) limit hyponym candidates within the entity that appears in the minibatch.
        if self._limit_hyponym_candidates_within_minibatch:
            set_candidates = set()
            for hyponymy in batch_hyponymy:
                set_candidates.add(hyponymy["hyponym"])
                set_candidates.add(hyponymy["hypernym"])
        else:
            set_candidates = None

        # create non-hyponymy relation
        lst_non_hyponymy_samples = []
        for hyponymy in batch_hyponymy:
            hyper = hyponymy["hypernym"]
            hypo = hyponymy["hyponym"]
            pos = hyponymy.get("pos", None)
            lst_tup_sample_b = []
            if "hyponym" in self._non_hyponymy_relation_target:
                lst_tup_sample_b_swap_hypo = self._taxonomy.sample_random_hyponyms(entity=hyper, candidates=set_candidates, size=size_per_sample,
                                                            exclude_hypernyms=self._exclude_reverse_hyponymy_from_non_hyponymy_relation,
                                                            part_of_speech=pos)
                lst_tup_sample_b.extend(lst_tup_sample_b_swap_hypo)
            if "hypernym" in self._non_hyponymy_relation_target:
                lst_tup_sample_b_swap_hyper = self._taxonomy.sample_random_hypernyms(entity=hypo, candidates=set_candidates, size=size_per_sample,
                                                            exclude_hypernyms=self._exclude_reverse_hyponymy_from_non_hyponymy_relation,
                                                            part_of_speech=pos)
                lst_tup_sample_b.extend(lst_tup_sample_b_swap_hyper)

            # convert them to dictionary format
            keys = ("hypernym", "hyponym", "distance")
            lst_non_hyponymy_samples.extend([dict(zip(keys, values)) for values in lst_tup_sample_b])

            if self._verbose:
                if len(lst_tup_sample_b):
                    f"failed to sample hyponyms: {hyper}"

        return lst_non_hyponymy_samples

    def _create_swap_hyponymy_samples(self, batch_hyponymy: List[Dict[str,Union[str,float]]]):
        lst_swap_hyponymy_samples = []
        for hyponymy in batch_hyponymy:
            hyper_rev = hyponymy["hyponym"]
            hypo_rev = hyponymy["hypernym"]
            dist_orig = hyponymy["distance"]
            pos = hyponymy.get("pos", None)

            if dist_orig == 1.0:
                dist_rev = -1.0
            else:
                is_hyponymy_relation = self._taxonomy.is_hyponymy_relation(hypernym=hyper_rev, hyponym=hypo_rev, part_of_speech=pos,
                                                                           include_reverse_hyponymy=False, not_exists=None)
                if is_hyponymy_relation is None:
                    continue
                elif is_hyponymy_relation:
                    dist_rev = 1.0
                else:
                    dist_rev = -1.0

            d = {
                "hyponym":hypo_rev,
                "hypernym":hyper_rev,
                "distance":dist_rev
            }
            lst_swap_hyponymy_samples.append(d)

        return lst_swap_hyponymy_samples

    def _create_batch_from_hyponymy_samples(self, batch_hyponymy: List[Dict[str,Union[str,float]]]):

        # take tokens from hyponymy pairs
        set_tokens = set()
        iter_hyponyms = (hyponymy["hyponym"] for hyponymy in batch_hyponymy)
        iter_hypernyms = (hyponymy["hypernym"] for hyponymy in batch_hyponymy)
        set_tokens.update(iter_hyponyms)
        set_tokens.update(iter_hypernyms)

        # take remaining tokens randomly from word embeddings dataset vocabulary
        n_diff = self._embedding_batch_size - len(set_tokens)
        if n_diff > 0:
            lst_index = np.random.randint(low=0, high=self._n_embeddings, size=n_diff)
            lst_tokens_from_embeddings = self._word_embeddings_dataset.indices_to_entities(lst_index)
            # add to token set
            set_tokens.update(lst_tokens_from_embeddings)
        else:
            lst_tokens_from_embeddings = []

        # create temporary token-to-index mapping
        lst_tokens = list(set_tokens)
        token_to_index = {token:idx for idx, token in enumerate(lst_tokens)}

        # create embeddings
        embeddings = tuple(self._word_embeddings_dataset[token]["embedding"] for token in lst_tokens)
        n_dim = self._word_embeddings_dataset.n_dim
        mat_embeddings = np.concatenate(embeddings, axis=0).reshape(-1, n_dim)

        # create hyponymy relations
        iter_idx_hypo = map(token_to_index.get, (hyponymy["hyponym"] for hyponymy in batch_hyponymy))
        iter_idx_hyper = map(token_to_index.get, (hyponymy["hypernym"] for hyponymy in batch_hyponymy))
        ## hyponymy score (originally, distance between hypernym and hyponym)
        iter_hyponymy_score = (hyponymy["distance"] for hyponymy in batch_hyponymy)
        ## convert hyponymy score into entailment probability
        ## if hyponymy score is greater than 1.0, then entailment probability is 1.0. otherwise, 0.0.
        iter_entailment_probability = (1.0 if hyponymy_score >= 1.0 else 0.0 for hyponymy_score in iter_hyponymy_score)

        lst_hyponymy_relation = [tup for tup in zip(iter_idx_hyper, iter_idx_hypo, iter_entailment_probability)]

        batch = {
            "embedding": mat_embeddings,
            "entity": lst_tokens,
            "hyponymy_relation": lst_hyponymy_relation,
            "hyponymy_relation_raw": batch_hyponymy
        }
        if self._verbose:
            batch["entity_sampled"] = lst_tokens_from_embeddings

        return batch

    def __getitem__(self, idx):

        while True:
            n_idx_min = self._hyponymy_batch_size * idx
            n_idx_max = self._hyponymy_batch_size * (idx+1)

            # feed hyponymy relation from the hyponymy dataset
            idx_hyponymy = self._sample_order[n_idx_min:n_idx_max]
            batch_hyponymy_b = self._hyponymy_dataset[idx_hyponymy]

            # remove hyponymy pairs which is not encodable
            batch_hyponymy = [sample for sample in batch_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]
            if len(batch_hyponymy) == 0:
                idx += 1
                continue

            # we randomly sample the non-hyponymy relation from the mini-batch
            n_adjuster = len(self._non_hyponymy_relation_target)
            size_per_sample = self._non_hyponymy_multiple // n_adjuster
            batch_non_hyponymy_b = self._create_non_hyponymy_samples_from_hyponymy_samples(batch_hyponymy=batch_hyponymy,
                                                                                         size_per_sample=size_per_sample)

            # remove hyponymy pairs which is not encodable
            batch_non_hyponymy = [sample for sample in batch_non_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]

            # concat it
            batch_hyponymy.extend(batch_non_hyponymy)

            # (optional) create swapped samples
            if self._swap_hyponymy_relations:
                batch_hyponymy_swap = self._create_swap_hyponymy_samples(batch_hyponymy=batch_hyponymy)
                batch_hyponymy.extend(batch_hyponymy_swap)

            # create (and format) a minibatch from both hyponymy samples and non-hyponymy samples
            batch = self._create_batch_from_hyponymy_samples(batch_hyponymy=batch_hyponymy)

            return batch

    def __iter__(self):

        for batch_hyponymy_b in self._hyponymy_dataloader:
            # remove hyponymy pairs which is not encodable
            batch_hyponymy = [sample for sample in batch_hyponymy_b if self.is_encodable_all(sample["hyponym"], sample["hypernym"])]
            if len(batch_hyponymy) == 0:
                continue

            # we randomly sample the non-hyponymy relation from the mini-batch
            n_adjuster = len(self._non_hyponymy_relation_target)
            size_per_sample = self._non_hyponymy_multiple // n_adjuster
            batch_non_hyponymy = self._create_non_hyponymy_samples_from_hyponymy_samples(batch_hyponymy=batch_hyponymy,
                                                                                         size_per_sample=size_per_sample)
            batch_hyponymy.extend(batch_non_hyponymy)

            # (optional) create swapped samples
            if self._swap_hyponymy_relations:
                batch_hyponymy_swap = self._create_swap_hyponymy_samples(batch_hyponymy=batch_hyponymy)
                batch_hyponymy.extend(batch_hyponymy_swap)

            # create (and format) a minibatch from both hyponymy samples and non-hyponymy samples
            batch = self._create_batch_from_hyponymy_samples(batch_hyponymy=batch_hyponymy)

            yield batch
