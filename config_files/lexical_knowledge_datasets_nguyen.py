#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from dataset.transform import FieldTypeConverter
from dataset.filter import DictionaryFilter
from .constants import SYNSETS_DEPTH_UP_TO_SECOND_LEVEL
_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})
_exclude_top_synsets = DictionaryFilter(excludes={"synset_hypernym":set(SYNSETS_DEPTH_UP_TO_SECOND_LEVEL)})
_exclude_synonymy_relations = DictionaryFilter(excludes={"distance":{0,"0"}})
# 上位下位関係のホップ数を制約 かつ synonymを除外
_include_direct_hyponymies = DictionaryFilter(includes={"distance":set("1")})
_include_hyponymies_up_to_three = DictionaryFilter(includes={"distance":set("123")})
_include_hyponymies_up_to_five = DictionaryFilter(includes={"distance":set("12345")})
_include_hyponymies_up_to_seven = DictionaryFilter(includes={"distance":set("1234567")})

DIR_LEXICAL_KNOWLEDGE = "/home/sakae/Windows/dataset/hypernym_detection/wordnet_nguyen_2017/"
DIR_LEXICAL_KNOWLEDGE_VER2 = "/home/sakae/Windows/dataset/hypernym_detection/wordnet_nguyen_2017_ver2/"

# lexical knowledge: graded hyponymy relations extracted from WordNet corpus.
# lemmas in these files are case sensitive and phrase included.
cfg_hyponymy_relation_datasets = {
    "WordNet-hyponymy-noun": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: noun",
    },
    "WordNet-hyponymy-noun-excl-top-synsets": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _exclude_top_synsets,
        "description": "WordNet-hyponymy relation dataset: noun, excluding top-2 synsets",
    },
    "WordNet-hyponymy-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: verb",
    },
    "WordNet-hyponymy-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: noun and verb",
    },
    "WordNet-hyponymy-noun-verb-excl-top-synsets": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _exclude_top_synsets,
        "description": "WordNet-hyponymy relation dataset: noun and verb, excluding top-2 noun synsets",
    },
    "WordNet-hyponymy-synonymy-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet hyponymy and synonymy relation dataset: noun and verb",
    },
    "WordNet-hyponymy-noun-verb-ver2": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _exclude_synonymy_relations,
        "description": "WordNet hyponymy relation dataset ver2(=excludes duplicate in xBLESS relation): noun and verb",
    },
    # 追試：ホップ数制約
    "WordNet-hyponymy-noun-verb-ver2-one_hop": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _include_direct_hyponymies,
        "description": "WordNet hyponymy relation dataset ver2(=excludes duplicate in xBLESS relation): noun and verb. limit hypponymy with n_hop=1.",
    },
    "WordNet-hyponymy-noun-verb-ver2-three_hop": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _include_hyponymies_up_to_three,
        "description": "WordNet hyponymy relation dataset ver2(=excludes duplicate in xBLESS relation): noun and verb. limit hypponymy with 1<=n_hop<=3.",
    },
    "WordNet-hyponymy-noun-verb-ver2-five_hop": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _include_hyponymies_up_to_five,
        "description": "WordNet hyponymy relation dataset ver2(=excludes duplicate in xBLESS relation): noun and verb. limit hypponymy with 1<=n_hop<=5.",
    },
        "WordNet-hyponymy-noun-verb-ver2-seven_hop": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE_VER2, "lexical_knowledge_wordnet_hyponymy_synonymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _include_hyponymies_up_to_seven,
        "description": "WordNet hyponymy relation dataset ver2(=excludes duplicate in xBLESS relation): noun and verb. limit hypponymy with 1<=n_hop<=5.",
    }
}

