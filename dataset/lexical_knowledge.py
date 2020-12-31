#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
from typing import Dict, List, Union, Callable, Optional, Any, Iterable
import torch
from torch.utils.data import Dataset
from _collections import defaultdict


class HyponymyDataset(Dataset):

    def __init__(self, path: str, header: bool, delimiter: str, columns: Dict[str, Union[int, slice]],
                 lowercase: bool = False,
                 replace_whitespace_with_underscore: bool = False,
                 description: str = "", transform=None, filter=None):

        super().__init__()
        self.path = path
        assert os.path.exists(path), f"invalid path specified: {path}"

        self._header = header
        self._delimiter = delimiter
        self._columns = columns
        self._lowercase = lowercase
        self._replace_whitespace = replace_whitespace_with_underscore
        self.description = description
        self.transform = transform
        self.filter = filter
        self._lst_samples = self._text_loader(path, header)
        if filter is not None:
            print(f"filter in valid samples...")
            self._lst_samples = self._filter_samples(self._lst_samples, filter)

    def _text_loader(self, path: str, header: bool):
        with io.open(path, mode="r") as ifs:
            if header:
                _ = next(ifs)
            ret = list(filter(bool, [record.strip() for record in ifs]))
        return ret

    def _filter_samples(self, lst_samples, filter_function: Callable[[Any], bool]):
        ret = []
        for s_entry in lst_samples:
            entry = self._preprocess(s_entry)
            if self.transform is not None:
                entry = self.transform(entry)

            if filter_function(entry):
                ret.append(s_entry)
        return ret

    def _apply(self, apply_column_name: str, apply_function: Callable, default_return_value: Optional[Any] = None):
        if apply_column_name not in self._columns:
            return default_return_value

        _transform_cache = self.transform
        self.transform = None

        it = (entry.get(apply_column_name, None) for entry in self)
        ret =  apply_function(filter(bool, it))

        self.transform = _transform_cache
        return ret

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
        return self._apply(apply_column_name="relation", apply_function=lambda it: list(set(it)), default_return_value=[])

    @property
    def classification_labels(self) -> List[str]:
        return self._apply(apply_column_name="is_hyponymy", apply_function=lambda it: list(set(it)), default_return_value=[])

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
    def max_distance(self):
        return self._apply(apply_column_name="distance", apply_function=lambda it: max(map(int,it)), default_return_value=None)

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
        return len(self._lst_samples)

    def _preprocess(self, s_entry: str):
        if self._lowercase:
            s_entry = s_entry.lower()
        lst_entry = s_entry.split(self._delimiter)
        if self._replace_whitespace:
            lst_entry = [entry.replace(" ","_") for entry in lst_entry]

        entry = {field_name:lst_entry[field_indices] for field_name, field_indices in self._columns.items()}
        return entry

    def __iter__(self):
        for s_entry in self._lst_samples:
            entry = self._preprocess(s_entry)
            if self.transform is not None:
                entry = self.transform(entry)
            yield entry

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, slice]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # index to entry (entries)
        if isinstance(idx, int):
            s_entry = self._lst_samples[idx]
        elif isinstance(idx, list):
            s_entry = [self._lst_samples[i] for i in idx]
        elif isinstance(idx, slice):
            s_entry = self._lst_samples[idx]
        else:
            raise NotImplementedError(f"unsupported index type: {type(idx)}")

        # entry (entries) to instances
        if isinstance(s_entry, str):
            entry = self._preprocess(s_entry)
            if self.transform is not None:
                entry = self.transform(entry)
        elif isinstance(s_entry, list):
            entry = map(self._preprocess, s_entry)
            if self.transform is not None:
                entry = map(self.transform, entry)
            entry = list(entry)

        return entry

    @property
    def verbose(self):
        ret = {
            "path": self.path,
            "n_sample": self.__len__(),
            "description": self.description,
            "transform": self.transform,
            "filter": self.filter
        }
        return ret

    def subset(self, indices: Iterable[int]):
        self._lst_samples = [self._lst_samples[idx] for idx in indices]


class WordNetHyponymyDataset(HyponymyDataset):
    pass


class WordNetSynsetHyponymyDataset(Dataset):

    def __init__(self, hyponymy_dataset: HyponymyDataset):

        self._hyponymy_dataset = hyponymy_dataset
        self._lst_samples = self._transform_to_synset_pairs(hyponymy_dataset)

    def _transform_to_synset_pairs(self, hyponymy_dataset: HyponymyDataset) -> List[Dict[str, Union[str, float]]]:
        dict_lemmas_temp = defaultdict(list)
        dict_lemmas = {}
        set_processed_synset_pairs = set()
        lst_records = []

        for record in hyponymy_dataset:
            synset_hyper = record["synset_hypernym"]
            synset_hypo = record["synset_hyponym"]
            lemma_hyper = record["hypernym"]
            lemma_hypo = record["hyponym"]

            dict_lemmas_temp[synset_hyper].append(lemma_hyper)
            dict_lemmas_temp[synset_hypo].append(lemma_hypo)

        # drop duplicates
        for synset, lst_lemma in dict_lemmas_temp.items():
            dict_lemmas[synset] = list(set(lst_lemma))

        for record in hyponymy_dataset:
            synset_hyper = record["synset_hypernym"]
            synset_hypo = record["synset_hyponym"]
            distance = record["distance"]
            pos = record["pos"]

            synset_pair = (synset_hyper, synset_hypo)
            if synset_pair in set_processed_synset_pairs:
                continue

            record_new = {
                "hypernym":synset_hyper,
                "hyponym":synset_hypo,
                "lemma_hypernym":dict_lemmas[synset_hyper],
                "lemma_hyponym":dict_lemmas[synset_hypo],
                "pos":pos,
                "distance":distance
            }
            lst_records.append(record_new)
            set_processed_synset_pairs.add(synset_pair)

        return lst_records

    def __len__(self):
        return len(self._lst_samples)

    def __iter__(self):
        for entry in self._lst_samples:
            yield entry

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, slice]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # index to entry (entries)
        if isinstance(idx, int):
            entry = self._lst_samples[idx]
        elif isinstance(idx, list):
            entry = [self._lst_samples[i] for i in idx]
        elif isinstance(idx, slice):
            entry = self._lst_samples[idx]
        else:
            raise NotImplementedError(f"unsupported index type: {type(idx)}")

        return entry

    @property
    def verbose(self):
        ret = self._hyponymy_dataset.verbose
        ret["n_sample"] = self.__len__()
        return ret

    def subset(self, indices: Iterable[int]):
        self._lst_samples = [self._lst_samples[idx] for idx in indices]
