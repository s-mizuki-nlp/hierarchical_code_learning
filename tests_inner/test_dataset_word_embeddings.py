#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import sys, io, os
from typing import List, Dict, Optional
from collections.abc import Callable
import unittest
from data.word_embeddings import Word2VecDataset, FastTextDataset
from config_files.word_embeddings import DIR_WORD_EMBEDDINGS

class BaseValidator(unittest.TestCase):

    def check_property_availability(self, object, target_property_names, object_name=""):

        for property_name in target_property_names:
            with self.subTest(object_name=object_name, property_name=property_name):
                self.assertTrue(hasattr(object, property_name))

    def check_method_availabilty(self, object, target_method_names, object_name=""):

        for method_name in target_method_names:
            with self.subTest(object_name=object_name, method_name=method_name):
                if method_name == "__call__" and isinstance(object, type):
                    self.assertTrue(issubclass(object, Callable))
                else:
                    self.assertTrue(callable(getattr(object, method_name, None)))

    def check_availability(self, test_object: object, test_object_name: str = "",
                           target_property_names: Optional[List[str]] = None, target_method_names: Optional[List[str]] = None):

        target_property_names = [] if target_property_names is None else target_property_names
        target_method_names = [] if target_method_names is None else target_method_names

        self.check_method_availabilty(test_object, target_method_names, test_object_name)
        self.check_property_availability(test_object, target_property_names, test_object_name)


class ImplementationTestCases(BaseValidator, unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._test_objects = {
            "Word2VecDataset": Word2VecDataset,
            "FastTextDataset": FastTextDataset
        }

    def test_property_method_availability(self):

        test_objects = self._test_objects
        target_property_names = "n_dim,vocab".split(",")
        target_method_names  = "encode,encode_phrase,is_encodable,index_to_entity".split(",")

        for obj_name, obj in test_objects.items():
            self.check_availability(obj, obj_name, target_property_names, target_method_names)


class Word2VecTestCases(unittest.TestCase):

    _cfg_embeddings = {
        "path_word2vec_format":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-google-news-300/word2vec-google-news-300"),
        "binary":True,
        "init_sims":False,
        "transform":None,
        "enable_phrase_composition":True
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls._dataset = Word2VecDataset(**cls._cfg_embeddings)

    def test_item_key_consistency(self):

        idx = 5
        entity = self._dataset.index_to_entity(idx)

        record_i = self._dataset[idx]
        record_e = self._dataset[entity]

        self.assertEqual(record_i["entity"], record_e["entity"])
        self.assertTrue(np.array_equal(record_i["embedding"], record_e["embedding"]))

    def test_phrase_splitter(self):

        phrase = "New_York_Times"
        lst_tokens = ["New_York","Times"]
        self.assertFalse(phrase in self._dataset.vocab)
        self.assertTrue(all(token in self._dataset.vocab for token in lst_tokens))

        lst_tokens_pred = self._dataset.phrase_splitter(phrase)
        self.assertEqual(lst_tokens, lst_tokens_pred)

    def test_phrase_composition(self):

        phrase = "max_margin_principle"
        lst_tokens = self._dataset.phrase_splitter(phrase)
        self.assertTrue(self._dataset.is_encodable(phrase))

        vec_e = self._dataset.encode(phrase)
        vec_e_pred = np.sum([self._dataset.encode(token) for token in lst_tokens], axis=0) / len(lst_tokens)

        self.assertTrue(np.allclose(vec_e, vec_e_pred))