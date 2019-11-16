#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Dict, Callable
import os, sys, io
import numpy as np

class EmbeddingNormalizer(object):

    def __init__(self, field_name_embedding="embedding"):
        self._field_name = field_name_embedding

    def __call__(self, sample):

        x = sample[self._field_name]
        sample[self._field_name] = x / np.linalg.norm(x)

        return sample


class HyponymyEntryToListOfHyponymyPair(object):

    def __init__(self, field_name_hyponym: str = "hyponym", field_name_hypernym: str = "hypernym", field_name_hypernyms: str = "hypernyms"):
        self._field_name_hyponym = field_name_hyponym
        self._field_name_hypernym = field_name_hypernym
        self._field_name_hypernyms = field_name_hypernyms

    def __call__(self, sample):
        hyponym = sample.get(self._field_name_hyponym, None)
        hypernym = sample.get(self._field_name_hypernym, None)
        hypernyms = sample.get(self._field_name_hypernyms, None)

        assert hyponym is not None, "could not get hyponym."

        if hypernym is not None:
            return [(hyponym, hypernym)]
        elif isinstance(hypernyms, list):
            return [(hyponym, hyper) for hyper in hypernyms]
        else:
            raise AssertionError("could not get hypernym.")


class FieldTypeConverter(object):

    def __init__(self, dict_field_type_converter: Dict[str, Callable]):

        self._dict_field_type_converter = dict_field_type_converter

    def __call__(self, sample):

        for field_name, converter in self._dict_field_type_converter.items():
            if field_name in sample:
                sample[field_name] = converter(sample[field_name])

        return sample