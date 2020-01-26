#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("ignore", category=ImportWarning)

import numpy as np
from torch.utils.data import DataLoader
import sys, io, os
import unittest
from dataset.lexical_knowledge import HyponymyDataset
from dataset.word_embeddings import Word2VecDataset, GeneralPurposeEmbeddingsDataset
from dataset.transform import FieldTypeConverter
from dataset.embeddings_plus_lexical_knowledge import WordEmbeddingsAndHyponymyDataset, WordEmbeddingsAndHyponymyDatasetWithNonHyponymyRelation

from config_files.word_embeddings import DIR_WORD_EMBEDDINGS
from config_files.lexical_knowledge_datasets import DIR_LEXICAL_KNOWLEDGE
from config_files.toy_dataset_ver2 import DIR_TOY_DATASET

_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})

class WordEmbeddingsAndHyponymyDatasetTestCases(unittest.TestCase):

    _cfg_embeddings = {
        "path_word2vec_format":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-google-news-300/word2vec-google-news-300"),
        "binary":True,
        "init_sims":False,
        "transform":None,
        "enable_phrase_composition":True
    }
    _cfg_lexical_knowledge = {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "transform": _distance_str_to_float,
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos": 3},
        "description": "WordNet-hyponymy relation dataset: noun and verb",
    }

    @classmethod
    def setUpClass(cls) -> None:

        cls._word_embeddings = Word2VecDataset(**cls._cfg_embeddings)
        cls._lexical_knowledge = HyponymyDataset(**cls._cfg_lexical_knowledge)
        cls._dataset = WordEmbeddingsAndHyponymyDataset(cls._word_embeddings, cls._lexical_knowledge,
                                                        embedding_batch_size=10, hyponymy_batch_size=2, verbose=True, shuffle=True)
        cls._dataloader = DataLoader(cls._dataset, batch_size=None, collate_fn=lambda v: v)

        idx_max = len(cls._dataset) - 1
        cls._batch = cls._dataset[idx_max // 2]

    def test_batch_fields(self):

        batch = self._batch
        # batch = next(iter(self._dataloader))
        required_fieid_names = "embedding,entity,hyponymy_relation".split(",")
        for field_name in required_fieid_names:
            with self.subTest(field_name=field_name):
                self.assertTrue(field_name in batch)

    def test_batch_field_size(self):

        batch = self._batch
        # batch = next(iter(self._dataloader))
        embedding = batch["embedding"]
        entity = batch["entity"]
        hyponymy_relation = batch["hyponymy_relation"]

        n_entity = len(entity)
        n_hyponymy_relation = len(hyponymy_relation)
        n_dim = self._word_embeddings.n_dim
        self.assertEqual(embedding.shape, (n_entity, n_dim))
        self.assertGreaterEqual(10, n_entity)
        self.assertGreaterEqual(2, n_hyponymy_relation)

    def test_batch_entity_consistency(self):

        batch = self._batch
        # batch = next(iter(self._dataloader))
        embedding = batch["embedding"]
        entity = batch["entity"]
        hyponymy_relation = batch["hyponymy_relation"]
        hyponymy_relation_raw = batch["hyponymy_relation_raw"]

        for dict_hyponymy in hyponymy_relation_raw:
            hyponym, hypernym, distance = dict_hyponymy["hyponym"], dict_hyponymy["hypernym"], dict_hyponymy["distance"]
            if (hyponym in entity) and (hypernym in entity):
                idx_hypo = entity.index(hyponym)
                idx_hyper = entity.index(hypernym)

                with self.subTest(hyponym=hyponym):
                    vec_e = embedding[idx_hypo,:]
                    vec_e_gt = self._word_embeddings[hyponym]["embedding"]
                    self.assertTrue(np.array_equal(vec_e, vec_e_gt))

                with self.subTest(hypernym=hypernym):
                    vec_e = embedding[idx_hyper,:]
                    vec_e_gt = self._word_embeddings[hypernym]["embedding"]
                    self.assertTrue(np.array_equal(vec_e, vec_e_gt))

                with self.subTest(hyponym=hyponym, hypernym=hypernym):
                    entry = (idx_hyper, idx_hypo, distance)
                    found = any([entry == hyponymy for hyponymy in hyponymy_relation])
                    self.assertTrue(found)


class WordEmbeddingsAndHyponymyDatasetWithNonHyponymyRelationTestCases(unittest.TestCase):

    _cfg_embeddings = {
        "path_numpy_array_binary_format":os.path.join(DIR_TOY_DATASET, "embeddings.npy"),
        "path_vocabulary_text":os.path.join(DIR_TOY_DATASET, "vocabulary.txt"),
        "path_vocabulary_information_json":os.path.join(DIR_TOY_DATASET, "entities.json"),
        "transform":None,
        "enable_phrase_composition":False
    }
    _cfg_lexical_knowledge = {
        "path": os.path.join(DIR_TOY_DATASET, "hyponymy_triple_train.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "distance":2},
        "transform": _distance_str_to_float,
        "description": "toy dataset (trianset). n_sample=52345, distance=[-1,1,2,3]"
    }

    @classmethod
    def setUpClass(cls) -> None:

        cls._word_embeddings = GeneralPurposeEmbeddingsDataset(**cls._cfg_embeddings)
        cls._lexical_knowledge = HyponymyDataset(**cls._cfg_lexical_knowledge)
        cls._dataset = WordEmbeddingsAndHyponymyDatasetWithNonHyponymyRelation(cls._word_embeddings, cls._lexical_knowledge,
                                                        embedding_batch_size=40, hyponymy_batch_size=4,
                                                        non_hyponymy_batch_size=24,
                                                        non_hyponymy_relation_distance=None,
                                                        non_hyponymy_relation_target="hyponym,hypernym",
                                                        non_hyponymy_weighted_sampling=True,
                                                        exclude_reverse_hyponymy_from_non_hyponymy_relation=True,
                                                        limit_hyponym_candidates_within_minibatch=False,
                                                        split_hyponymy_and_non_hyponymy=True,
                                                        entity_depth_information="both",
                                                        verbose=True, shuffle=True)
        cls._dataloader = DataLoader(cls._dataset, batch_size=None, collate_fn=lambda v: v)

        idx_max = len(cls._dataset) - 1
        cls._batch = cls._dataset[idx_max // 2]

    def test_batch_fields(self):
        batch = self._batch
        # batch = next(iter(self._dataloader))
        required_fieid_names = "embedding,entity,hyponymy_relation,non_hyponymy_relation,hyponymy_relation_raw,non_hyponymy_relation_raw".split(",")
        for field_name in required_fieid_names:
            with self.subTest(field_name=field_name):
                self.assertTrue(field_name in batch)

    def test_batch_field_size(self):

        batch = self._batch
        embedding = batch["embedding"]
        entity = batch["entity"]
        hyponymy_relation = batch["hyponymy_relation"]
        non_hyponymy_relation = batch["non_hyponymy_relation"]

        n_entity = len(entity)
        n_hyponymy_relation = len(hyponymy_relation)
        n_non_hyponymy_relation = len(non_hyponymy_relation)
        n_dim = self._word_embeddings.n_dim
        self.assertEqual(embedding.shape, (n_entity, n_dim))
        self.assertGreaterEqual(self._dataset._embedding_batch_size, n_entity)
        self.assertGreaterEqual(self._dataset._hyponymy_batch_size, n_hyponymy_relation)
        self.assertGreaterEqual(self._dataset._non_hyponymy_batch_size, n_non_hyponymy_relation)

    def test_batch_entity_consistency(self):

        batch = self._batch
        embedding = batch["embedding"]
        entity = batch["entity"]
        hyponymy_relation = batch["hyponymy_relation"]
        hyponymy_relation_raw = batch["hyponymy_relation_raw"]

        for dict_hyponymy in hyponymy_relation_raw:
            hyponym, hypernym, distance = dict_hyponymy["hyponym"], dict_hyponymy["hypernym"], dict_hyponymy["distance"]
            if (hyponym in entity) and (hypernym in entity):
                idx_hypo = entity.index(hyponym)
                idx_hyper = entity.index(hypernym)

                with self.subTest(hyponym=hyponym):
                    vec_e = embedding[idx_hypo,:]
                    vec_e_gt = self._word_embeddings[hyponym]["embedding"]
                    self.assertTrue(np.array_equal(vec_e, vec_e_gt))

                with self.subTest(hypernym=hypernym):
                    vec_e = embedding[idx_hyper,:]
                    vec_e_gt = self._word_embeddings[hypernym]["embedding"]
                    self.assertTrue(np.array_equal(vec_e, vec_e_gt))

                with self.subTest(hyponym=hyponym, hypernym=hypernym):
                    entry = (idx_hyper, idx_hypo, distance)
                    found = any([entry == hyponymy for hyponymy in hyponymy_relation])
                    self.assertTrue(found)

    def test_batch_non_hyponymy_entity_consistency(self):
        # test if non-hyponymy relation does not appear in the hyponymy relation.

        batch = self._batch
        embedding = batch["embedding"]
        entity = batch["entity"]
        non_hyponymy_relation = batch["non_hyponymy_relation"] # list of (idx_hyper, idx_hypo, distance)
        taxonomy = self._dataset._taxonomy

        # hyponymy and reverse-hyponymy relation within the lexical knowledge dataset
        set_hyponymy_relation = set()
        for batch in self._lexical_knowledge:
            tup = (batch["hyponym"], batch["hypernym"])
            tup_rev = (batch["hypernym"], batch["hyponym"])
            set_hyponymy_relation.add(tup)
            set_hyponymy_relation.add(tup_rev)

        for idx_hyper, idx_hypo, distance in non_hyponymy_relation:
            hypernym = entity[idx_hyper]
            hyponym = entity[idx_hypo]
            tup_non_hyponymy = (hypernym, hyponym)
            tup_rev_non_hyponymy = (hyponym, hypernym)

            # embedding consistency
            with self.subTest(hyponym=hyponym):
                vec_e = embedding[idx_hypo,:]
                vec_e_gt = self._word_embeddings[hyponym]["embedding"]
                self.assertTrue(np.array_equal(vec_e, vec_e_gt))

            with self.subTest(hypernym=hypernym):
                vec_e = embedding[idx_hyper,:]
                vec_e_gt = self._word_embeddings[hypernym]["embedding"]
                self.assertTrue(np.array_equal(vec_e, vec_e_gt))

            # assert non-hyponymy relation never appear in the trainset
            with self.subTest(hyponym=hyponym, hypernym=hypernym):
                self.assertTrue(tup_non_hyponymy not in set_hyponymy_relation)
                self.assertTrue(tup_rev_non_hyponymy not in set_hyponymy_relation)

            # assert distance of the non-hyponymy relation
            with self.subTest(hyponym=hyponym, hypernym=hypernym):
                if self._dataset._non_hyponymy_relation_distance is None:
                    distance_gt = taxonomy.hyponymy_distance(hypernym=hypernym, hyponym=hyponym)
                else:
                    distance_gt = self._dataset._non_hyponymy_relation_distance
                self.assertEqual(distance, distance_gt)

    def test_entity_depth(self):

        batch = self._batch
        entity = batch["entity"]
        entity_depth = batch["entity_depth"]
        taxonomy = self._dataset._taxonomy

        for idx, depth in entity_depth:
            e = entity[idx]
            depth_gt = taxonomy.depth(e, offset=1)
            with self.subTest(entity=e):
                self.assertEqual(depth, depth_gt)
