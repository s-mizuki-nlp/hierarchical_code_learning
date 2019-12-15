#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import unittest
import networkx as nx
from dataset.utils import BasicTaxonomy, WordNetTaxonomy


class BasicTaxonomyTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        edges = """
        F,I
        F,H
        F,A
        A,B
        A,E
        B,G
        B,D
        E,D
        E,J
        J,M
        L,K
        K,C
        C,J
        N,O
        """

        cls._edges = [pair.strip().split(",") for pair in edges.strip().split("\n")]

        g = nx.DiGraph()
        g.add_edges_from(cls._edges)
        cls._graph = g

        taxonomy = BasicTaxonomy()
        taxonomy.build_directed_acyclic_graph(iter_hyponymy_pairs=cls._edges)
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
                pred = self._taxonomy.hyponymy_distance(hypernym=e1, hyponym=e2, not_exists=None)
                self.assertTrue(pred is None)

    def test_hyponymy_distance(self):

        test_cases = [("F","A",1),("A","D",2),("E","M",2),("M","E",-2),("D","A",-2)]
        test_cases.extend([("A","N",-2),("N","E",-1),("B","L",-3)])
        test_cases.extend([("B","E",-1),("B","J",-1),("J","B",-2),("B","H",-2),("G","J",-2)])

        for e1, e2, dist_gt in test_cases:
            with self.subTest(e1=e1, e2=e2, dist=dist_gt):
                pred = self._taxonomy.hyponymy_distance(hypernym=e1, hyponym=e2, not_exists=None)
                self.assertEqual(pred, dist_gt)

    def test_sample_non_hyponym_na(self):

        # impossible
        hypernym = "O"
        pred = self._taxonomy.sample_non_hyponym(entity=hypernym, candidates=["N","O"], exclude_hypernyms=True, size=1)
        self.assertEqual(len(pred), 0)

        # impossible (automatic)
        hypernym = "J"
        candidates = self._taxonomy.hyponyms(hypernym) | self._taxonomy.hypernyms(hypernym)
        pred = self._taxonomy.sample_non_hyponym(entity=hypernym, candidates=candidates, exclude_hypernyms=True, size=1)
        self.assertEqual(len(pred), 0)

        # possible, but only one choice
        hypernym = "O"
        pred = self._taxonomy.sample_non_hyponym(entity=hypernym, candidates=["N","O"], exclude_hypernyms=False, size=2)
        gt = ["N","N"]
        self.assertListEqual(pred, gt)

        # possible, but deterministic
        hypernym = "A"
        gt = set("I,H,L,K,C,N,O".split(","))
        pred = self._taxonomy.sample_non_hyponym(entity=hypernym, exclude_hypernyms=True, size=7)
        self.assertSetEqual(set(pred), gt)

    def test_sample_hyponymy_not_exclude_hypernyms(self):

        all_entities = set(self._taxonomy.dag.nodes)

        # check if random sample does not include hyponyms
        for entity in all_entities:
            gt = all_entities - self._taxonomy.hyponyms(entity) - set(entity)
            pred = self._taxonomy.sample_non_hyponym(entity, size=10, exclude_hypernyms=False)
            with self.subTest(hypernym=entity):
                self.assertTrue(set(pred).issubset(gt))

    def test_sample_hyponymy_exclude_hypernyms(self):

        all_entities = set(self._taxonomy.dag.nodes)

        # check if random sample does not include hyponyms and hypernyms
        for entity in all_entities:
            gt = all_entities - self._taxonomy.hyponyms(entity) - self._taxonomy.hypernyms(entity) - set(entity)
            pred = self._taxonomy.sample_non_hyponym(entity, size=10, exclude_hypernyms=True)
            with self.subTest(hypernym=entity):
                self.assertTrue(set(pred).issubset(gt))
