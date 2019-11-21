#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("always")

import numpy as np
from torch.utils.data import DataLoader
import sys, io, os
import unittest
from dataset.lexical_knowledge import HyponymyDataset
from dataset.word_embeddings import Word2VecDataset
from dataset.transform import FieldTypeConverter
from dataset.embeddings_plus_lexical_knowledge import WordEmbeddingsAndHyponymyDataset

from config_files.word_embeddings import DIR_WORD_EMBEDDINGS
from config_files.lexical_knowledge_datasets import DIR_LEXICAL_KNOWLEDGE

distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})

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
        "transform": distance_str_to_float,
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
                    entry = (idx_hypo, idx_hyper, distance)
                    found = any([entry == hyponymy for hyponymy in hyponymy_relation])
                    self.assertTrue(found)
