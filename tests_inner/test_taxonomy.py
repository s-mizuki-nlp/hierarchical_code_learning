#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import itertools
import tempfile
import unittest
import networkx as nx
import numpy as np
from dataset.taxonomy import BasicTaxonomy, WordNetTaxonomy
from dataset.lexical_knowledge import HyponymyDataset, WordNetHyponymyDataset
from dataset.transform import FieldTypeConverter
from config_files.lexical_knowledge_datasets_original import cfg_hyponymy_relation_datasets

_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})

class BasicTaxonomyTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        edges = """
        F,I,1
        F,H,1
        F,A,1
        A,B,1
        A,E,1
        B,G,1
        B,D,1
        E,D,1
        E,J,1
        J,M,1
        L,K,1
        K,C,1
        C,J,1
        N,O,1
        """

        cls._edges = [pair.strip().split(",")[:2] for pair in edges.strip().split("\n")]

        g = nx.DiGraph()
        g.add_edges_from(cls._edges)
        cls._graph = g

        # build taxonomy
        tmp = tempfile.NamedTemporaryFile(prefix="edges", mode="w")
        tmp.write(edges)
        tmp.flush()

        cfg = {
            "path":tmp.name,
            "header":False,
            "delimiter":",",
            "columns":{"hypernym":0, "hyponym":1, "distance":2},
            "lowercase":False,
            "transform":_distance_str_to_float
        }
        hyponymy_dataset = HyponymyDataset(**cfg)
        assert len(hyponymy_dataset) > 0
        tmp.close()

        BasicTaxonomy._DEBUG_MODE = True
        taxonomy = BasicTaxonomy(hyponymy_dataset=hyponymy_dataset)
        cls._taxonomy = taxonomy

    def test_depth(self):

        offset = 0
        test_cases = {
            "F":0, "A":1, "H":1, "D":3, "J":3, "C":2, "M":4, "N":0, "O":1
        }

        for entity, ground_truth in test_cases.items():
            with self.subTest(entity=entity):
                pred = self._taxonomy.depth(entity=entity, offset=offset)
                self.assertEqual(ground_truth, pred)

    def test_hypernyms(self):

        test_cases = {
            "F":set(),
            "A":set("F"),
            "D":set("F,A,B,E".split(",")),
            "M":set("F,A,E,J,C,K,L".split(",")),
            "O":set("N")
        }

        for entity, ground_truth in test_cases.items():
            with self.subTest(entity=entity):
                pred = self._taxonomy.hypernyms(entity=entity)
                self.assertSequenceEqual(ground_truth, pred)

    def test_hyponyms(self):

        test_cases = {
            "F":set("I,A,H,B,E,G,D,J,M".split(",")),
            "A":set("B,G,D,E,J,M".split(",")),
            "D":set(),
            "C":set("J,M".split(",")),
            "N":set("O")
        }

        for entity, ground_truth in test_cases.items():
            with self.subTest(entity=entity):
                pred = self._taxonomy.hyponyms(entity=entity)
                self.assertSequenceEqual(ground_truth, pred)


    def test_hyponymy_distance_na(self):

        test_cases = [("F","P"),("P","A"),("P","Q")]

        for e1, e2 in test_cases:
            with self.subTest(e1=e1, e2=e2):
                with self.assertRaises(Exception):
                    pred = self._taxonomy.hyponymy_score(hypernym=e1, hyponym=e2)

    def test_hyponymy_distance(self):

        test_cases = [("F","A",1),("A","D",2),("E","M",2),("M","E",-2),("D","A",-2)]
        test_cases.extend([("A","N",-2),("N","E",-1),("B","L",-3)])
        test_cases.extend([("B","E",-1),("B","J",-1),("J","B",-2),("B","H",-2),("G","J",-2)])

        for e1, e2, dist_gt in test_cases:
            with self.subTest(e1=e1, e2=e2, dist=dist_gt):
                pred = self._taxonomy.hyponymy_score(hypernym=e1, hyponym=e2)
                self.assertEqual(pred, dist_gt)

    def test_hyponymy_distance_all(self):
        # compare every possible combination with alternative implementation (=`hyponymy_score_slow()`)
        nodes = tuple(self._taxonomy.dag.nodes)
        for e1, e2 in itertools.product(nodes, nodes):
            if e1 == e2:
                continue
            gt = self._taxonomy.hyponymy_score_slow(e1, e2)
            pred = self._taxonomy.hyponymy_score(e1, e2)
            with self.subTest(hypernym=e1, hyponym=e2):
                self.assertEqual(gt, pred)

    def test_sample_non_hyponym_na(self):

        # impossible
        hypernym = "O"
        pred = self._taxonomy.sample_non_hyponymy(entity=hypernym, candidates=["N", "O"], exclude_hypernyms=True, size=1)
        self.assertEqual(len(pred), 0)

        # impossible (automatic)
        hypernym = "J"
        candidates = self._taxonomy.hyponyms(hypernym) | self._taxonomy.hypernyms(hypernym)
        pred = self._taxonomy.sample_non_hyponymy(entity=hypernym, candidates=candidates, exclude_hypernyms=True, size=1)
        self.assertEqual(len(pred), 0)

        # possible, but only one choice
        hypernym = "O"
        pred = self._taxonomy.sample_non_hyponymy(entity=hypernym, candidates=["N", "O"], exclude_hypernyms=False, size=2)
        gt = ("N","N")
        self.assertTupleEqual(pred, gt)

        # possible, but deterministic
        hypernym = "A"
        gt = set("I,H,L,K,C,N,O".split(","))
        pred = self._taxonomy.sample_non_hyponymy(entity=hypernym, exclude_hypernyms=True, size=7)
        self.assertTrue(set(pred).issubset(gt))

    def test_sample_hyponymy_not_exclude_hypernyms(self):

        all_entities = set(self._taxonomy.dag.nodes)

        # check if random sample does not include hyponyms
        for entity in all_entities:
            gt = all_entities - self._taxonomy.hyponyms(entity) - set(entity)
            pred = self._taxonomy.sample_non_hyponymy(entity, size=100, exclude_hypernyms=False)
            with self.subTest(hypernym=entity):
                self.assertTrue(set(pred).issubset(gt))

    def test_sample_hyponymy_exclude_hypernyms(self):

        all_entities = set(self._taxonomy.dag.nodes)

        # check if random sample does not include hyponyms and hypernyms
        for entity in all_entities:
            gt = all_entities - self._taxonomy.hyponyms(entity) - self._taxonomy.hypernyms(entity) - set(entity)
            pred = self._taxonomy.sample_non_hyponymy(entity, size=100, exclude_hypernyms=True)
            with self.subTest(hypernym=entity):
                self.assertTrue(set(pred).issubset(gt))


class WordNetTaxonomyTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        dataset = WordNetHyponymyDataset(**cfg_hyponymy_relation_datasets["WordNet-hyponymy-noun-verb"])
        taxonomy = WordNetTaxonomy(hyponymy_dataset=dataset)
        cls._taxonomy = taxonomy

    def test_sample_random_hypernyms_noun(self):
        entity = "cat"
        pos = "n"
        pred = self._taxonomy.sample_random_hypernyms(entity=entity, part_of_speech=pos, exclude_hypernyms=True)

        self.assertEqual(len(pred), 1)
        self.assertEqual(self._taxonomy.ACTIVE_ENTITY_TYPE, pos)

        hypernym, hyponym, score = pred[0]
        non_candidates = self._taxonomy.hypernyms_and_hyponyms_and_self(entity=entity)

        self.assertTrue(hypernym not in non_candidates)
        self.assertEqual(hyponym, entity)
        self.assertLess(score, 0)

    def test_sample_random_hyponyms_noun(self):
        entity = "cat"
        pos = "n"
        pred = self._taxonomy.sample_random_hyponyms(entity=entity, part_of_speech=pos, exclude_hypernyms=True)

        self.assertEqual(len(pred), 1)
        self.assertEqual(self._taxonomy.ACTIVE_ENTITY_TYPE, pos)

        hypernym, hyponym, score = pred[0]
        non_candidates = self._taxonomy.hypernyms_and_hyponyms_and_self(entity=entity)

        self.assertTrue(hyponym not in non_candidates)
        self.assertEqual(hypernym, entity)
        self.assertLess(score, 0)

    def test_sample_random_co_hyponyms_noun(self):
        hypernym = "carnivore"
        hyponym = "cat"
        pos = "n"
        size = 10
        lst_pred = self._taxonomy.sample_random_co_hyponyms(hypernym=hypernym, hyponym=hyponym, part_of_speech=pos, size=size)

        self.assertEqual(len(lst_pred), size)
        self.assertEqual(self._taxonomy.ACTIVE_ENTITY_TYPE, pos)

        non_candidates = self._taxonomy.hypernyms_and_hyponyms_and_self(entity=hyponym)
        for r_hypernym, r_hyponym, score in lst_pred:
            gt = self._taxonomy.hyponymy_score(hypernym=r_hypernym, hyponym=r_hyponym)

            with self.subTest(hypernym=r_hypernym, hyponym=r_hyponym):
                self.assertTrue(r_hypernym not in non_candidates)
                self.assertEqual(r_hyponym, hyponym)
                self.assertEqual(score, gt)

    def test_depth_noun(self):
        entity = "cat"
        pos = "n"
        pred = self._taxonomy.depth(entity=entity, part_of_speech=pos)
        gt = 10
        self.assertTrue(pred, gt)

    def test_hyponymy_scores(self):
        lst_tests = [
            {
                "hypernym":"carnivore",
                "hyponym":"cat",
                "pos":"n",
                "score":2
            },
            {
                "hypernym":"feel",
                "hyponym":"exuberate",
                "pos":"v",
                "score":3
            },
            {
                "hypernym":"regret",
                "hyponym":"exuberate",
                "pos":"v",
                "score":-1
            }
        ]

        for dict_test in lst_tests:
            gt = dict_test.pop("score")
            pos = dict_test.pop("pos")
            self._taxonomy.ACTIVE_ENTITY_TYPE = pos
            pred = self._taxonomy.hyponymy_score(**dict_test)

            with self.subTest(**dict_test):
                self.assertEqual(pred, gt)

    def test_cache_effect_depth(self):

        # validate cache is part-of-speech sensitive
        entity = "license"
        depth_n = self._taxonomy.depth(entity=entity, part_of_speech="n")
        depth_v = self._taxonomy.depth(entity=entity, part_of_speech="v")

        self.assertNotEqual(depth_n, depth_v)

    def test_cache_effect_hypernyms(self):

        # validate cache is part-of-speech sensitive
        entity = "license"
        self._taxonomy.ACTIVE_ENTITY_TYPE = "n"
        hyper_hypo_n = self._taxonomy.hypernyms_and_hyponyms_and_self(entity=entity)
        self._taxonomy.ACTIVE_ENTITY_TYPE = "v"
        hyper_hypo_v = self._taxonomy.hypernyms_and_hyponyms_and_self(entity=entity)

        self.assertNotEqual(hyper_hypo_n, hyper_hypo_v)