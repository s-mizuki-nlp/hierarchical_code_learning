#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
import argparse

import numpy as np

class EmbeddingNormalizer(object):

    def __call__(self, sample):

        x = sample["embedding"]
        sample["embedding"] = x / np.linalg.norm(x)

        return sample


class HyponymyEntryToListOfHyponymyPair(object):

    def __call__(self, sample):
        hyponym = sample.get("hyponym", None)
        hypernym = sample.get("hypernym", None)
        hypernyms = sample.get("hypernyms", None)

        assert hyponym is not None, "could not get hyponym."

        if hypernym is not None:
            return [(hyponym, hypernym)]
        elif isinstance(hypernyms, list):
            return [(hyponym, hyper) for hyper in hypernyms]
        else:
            raise AssertionError("could not get hypernym.")
