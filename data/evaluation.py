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