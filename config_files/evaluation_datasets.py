#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.transform import FieldTypeConverter
from dataset.filter import DictionaryFilter
from .constants import HYPERLEX_HYPONYMY_SYNONYMY_RELATIONS

DIR_EVALSET = "/home/sakae/Windows/dataset/hypernym_detection/evalset_nguyen_2017/"

# evaluation dataset for ranking retrieval task
cfg_evaluation_datasets_ranking_retrieval = {
    "BLESS": { # case sensitive
        "path": os.path.join(DIR_EVALSET, "ranking_retrieval/BLESS.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "description": "BLESS dataset: ranking retrieval",
    },
    "EVALution": {
        "path": os.path.join(DIR_EVALSET, "ranking_retrieval/EVALution_exclude_entailm.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "description": "EVALution dataset: ranking retrieval",
    },
    "Lench&Benotto": { # case sensitive
        "path": os.path.join(DIR_EVALSET, "ranking_retrieval/LenciBenotto.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "description": "Lench&Benotto dataset: ranking retrieval",
    },
    "Weeds": {
        "path": os.path.join(DIR_EVALSET, "ranking_retrieval/Weeds.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "description": "Weeds dataset: ranking retrieval",
    }
}

def _relation_to_direction_class(sample):
    class_mapper = {"hyper":"x"}
    sample["class"] = class_mapper[sample["relation"]]
    return sample

def _relation_to_binary_class_wbless(sample):
    class_mapper = {"hyper":True, "other":False}
    sample["class"] = class_mapper.get(sample["relation"], False)
    return sample

def _relation_to_binary_class_entailment(sample):
    class_mapper = {"hyper":True, "random":False}
    sample["class"] = class_mapper.get(sample["relation"], False)
    return sample

def _relation_to_three_classes(sample):
    class_mapper = {"hyper":"hyponymy", "rhyper":"reverse-hyponymy", "other":"other"}
    sample["class"] = class_mapper.get(sample["relation"], "other")
    return sample

_rating_str_to_float = FieldTypeConverter(dict_field_type_converter={"rating":float})

# evaluation dataset for classification task
# ENTAILMENT dataset は optional. 除外してもよい
cfg_evaluation_datasets_classification = {
    "BLESS-hyponymy": {
        "path": os.path.join(DIR_EVALSET, "classification/BLESS_hyponymy.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "transform":_relation_to_direction_class,
        "description": "BLESS-hyponymy dataset: directionality",
    },
    "WBLESS": {
        "path": os.path.join(DIR_EVALSET, "classification/AWBLESS.txt"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "transform":_relation_to_binary_class_wbless,
        "description": "Weeds-BLESS dataset: hyponymy or not",
    },
    "WBLESS-ext": {
        "path": os.path.join(DIR_EVALSET, "classification/AWBLESS.txt.ext"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3, "fine_grained_relation": 4},
        "transform":_relation_to_binary_class_wbless,
        "description": "Weeds-BLESS dataset with BLESS fine-grained relation: hyponymy or not",
    },
    "BIBLESS": {
        "path": os.path.join(DIR_EVALSET, "classification/ABIBLESS.txt"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "transform":_relation_to_three_classes,
        "description": "BIBLESS relabeled dataset: hyponymy, reverse-hyponymy or other",
    },
    "BIBLESS-ext": {
        "path": os.path.join(DIR_EVALSET, "classification/ABIBLESS.txt.ext"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3, "fine_grained_relation": 4},
        "transform":_relation_to_three_classes,
        "description": "BIBLESS relabeled dataset with BLESS fine-grained relation: hyponymy, reverse-hyponymy or other",
    },
    "ENTAILMENT": {
        "path": os.path.join(DIR_EVALSET, "classification_optional/Baroni2012.all"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "is_hyponymy":2, "relation": 3},
        "transform":_relation_to_binary_class_entailment,
        "description": "ENTAILMENT dataset[Baroni+ 2012]: hyponymy or not",
    }
}

# graded lexical entailment task

## custom filters
_include_noun = DictionaryFilter(includes={"pos-tag":{"N"}})
_include_hyponymy_and_synonymy = DictionaryFilter(includes={"relation":set(HYPERLEX_HYPONYMY_SYNONYMY_RELATIONS)})

cfg_evaluation_datasets_graded_le = {
    "HyperLex": {
        "path": os.path.join(DIR_EVALSET, "graded_lexical_entailment/hyperlex-all.txt"),
        "header": True,
        "delimiter": " ",
        "transform":_rating_str_to_float,
        "columns": {"hyponym":0, "hypernym":1, "pos-tag":2, "relation": 3, "rating": 4},
        "description": "HyperLex dataset: graded lexical entailment",
    },
    "HyperLex-noun": {
        "path": os.path.join(DIR_EVALSET, "graded_lexical_entailment/hyperlex-all.txt"),
        "header": True,
        "delimiter": " ",
        "transform":_rating_str_to_float,
        "filter":_include_noun,
        "columns": {"hyponym":0, "hypernym":1, "pos-tag":2, "relation": 3, "rating": 4},
        "description": "HyperLex dataset: noun word pairs",
    },
    "HyperLex-hyponymy-synonymy": {
        "path": os.path.join(DIR_EVALSET, "graded_lexical_entailment/hyperlex-all.txt"),
        "header": True,
        "delimiter": " ",
        "transform":_rating_str_to_float,
        "filter":_include_hyponymy_and_synonymy,
        "columns": {"hyponym":0, "hypernym":1, "pos-tag":2, "relation": 3, "rating": 4},
        "description": "HyperLex dataset: hyponymy, reverse-hyponymy, and synonymy relation word pairs",
    }
}

# ---- #

# testset for Hypernym discovery task
DIR_HYPERNYM_DISCOVERY_TRAIN_TEST = "/home/sakae/Windows/dataset/hypernym_detection/semeval_2018_hypernym_discovery/train_and_test"

cfg_evaluation_datasets_hypernym_discovery = {
    "HypernymDiscovery-1A": {
        "path": os.path.join(DIR_HYPERNYM_DISCOVERY_TRAIN_TEST, "test/gold/1A.english.test.gold.txt"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernyms":slice(1,None)},
        "replace_whitespace_with_underscore": True,
        "description": "Hypernym Discovery testset: 1A, English-General"
    },
    "HypernymDiscovery-2A": {
        "path": os.path.join(DIR_HYPERNYM_DISCOVERY_TRAIN_TEST, "test/gold/2A.medical.test.gold.txt"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernyms":slice(1,None)},
        "replace_whitespace_with_underscore": True,
        "description": "Hypernym Discovery testset: 2A, English-Medical"
    },
    "HypernymDiscovery-2B": {
        "path": os.path.join(DIR_HYPERNYM_DISCOVERY_TRAIN_TEST, "test/gold/2B.music.test.gold.txt"),
        "header": False,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernyms":slice(1,None)},
        "replace_whitespace_with_underscore": True,
        "description": "Hypernym Discovery testset: 2B, English-Music"
    }
}