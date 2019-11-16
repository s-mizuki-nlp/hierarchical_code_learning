#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
from typing import Dict, List, Union
import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
import fasttext
from gensim.models import KeyedVectors
from wikipedia2vec import Wikipedia2Vec


class HyponymyDataset(IterableDataset):

    def __init__(self, path: str, header: bool, delimiter: str, columns: Dict[str, Union[int, slice]],
                 lowercase: bool = False,
                 replace_whitespace_with_underscore: bool = False,
                 description: str = "", transform=None):

        super(HyponymyDataset).__init__()
        self.path = path
        assert os.path.exists(path), f"invalid path specified: {path}"

        self._header = header
        self._delimiter = delimiter
        self._columns = columns
        self._lowercase = lowercase
        self._replace_whitespace = replace_whitespace_with_underscore
        self._n_sample = None
        self.description = description
        self.transform = transform

    def _extract_distinct_values(self, column_name):
        if column_name not in self._columns:
            return []

        _transform_cache = self.transform
        self.transform = None

        set_entry = set()
        for entry in self:
            set_entry.add(entry.get(column_name, None))

        distinct_values = list(filter(bool, list(set_entry)))

        self.transform = _transform_cache
        return distinct_values

    def _test_case_sensitive(self, column_name):
        if column_name not in self._columns:
            return False

        _transform_cache = self.transform
        self.transform = None

        def is_case_sensitive(str_: str):
            return any(s.isupper() for s in str_)

        ret = any([is_case_sensitive(entry.get(column_name,"")) for entry in self])
        self.transform = _transform_cache
        return ret

    def _test_has_phrase(self, column_name):
        if column_name not in self._columns:
            return False

        _transform_cache = self.transform
        self.transform = None

        def has_phrase(str_: str):
            delimiters = (" ","_")
            return any(str_.find(delimiter) != -1 for delimiter in delimiters)

        ret = any([has_phrase(entry.get(column_name,"")) for entry in self])
        self.transform = _transform_cache
        return ret

    @property
    def relations(self) -> List[str]:
        return self._extract_distinct_values("relation")

    @property
    def classification_labels(self) -> List[str]:
        return self._extract_distinct_values("is_hyponymy")

    @property
    def is_case_sensitive(self):
        if self._lowercase:
            ret = False
        else:
            is_hypo_case_sensitive = self._test_case_sensitive("hyponym")
            is_hyper_case_sensitive = self._test_case_sensitive("hypernym")
            ret = is_hypo_case_sensitive | is_hyper_case_sensitive
        return ret

    @property
    def has_phrase(self):
        if self._lowercase:
            ret = False
        else:
            has_hypo_phrase = self._test_has_phrase("hyponym")
            has_hyper_phrase = self._test_has_phrase("hypernym")
            ret = has_hypo_phrase | has_hyper_phrase
        return ret

    @property
    def lowercase(self):
        return self._lowercase

    def __len__(self):
        if self._n_sample is not None:
            return self._n_sample
        else:
            n_sample = 0
            for _ in self:
                n_sample += 1
            self._n_sample = n_sample
            return self._n_sample

    def __iter__(self):

        ifs = io.open(self.path, mode="r")
        if self._header:
            _ = next(ifs)

        for s_entry in ifs:
            if self._lowercase:
                s_entry = s_entry.lower()
            lst_entry = s_entry.strip().split(self._delimiter)
            if self._replace_whitespace:
                lst_entry = [entry.replace(" ","_") for entry in lst_entry]

            entry = {field_name:lst_entry[field_indices] for field_name, field_indices in self._columns.items()}
            if self.transform is not None:
                entry = self.transform(entry)
            yield entry

        ifs.close()

    @property
    def verbose(self):
        ret = {
            "path": self.path,
            "n_sample": self.__len__(),
            "description": self.description,
            "transform": self.transform
        }
        return ret