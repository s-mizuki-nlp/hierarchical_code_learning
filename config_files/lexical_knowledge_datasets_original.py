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
from config_files.wordnet_constant import SYNSETS_DEPTH_UP_TO_THIRD_LEVEL

_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})
_exclude_top_synsets = DictionaryFilter(excludes={"synset_hypernym":set(SYNSETS_DEPTH_UP_TO_SECOND_LEVEL)})
# _exclude_top_synsets = DictionaryFilter(excludes={"synset_hypernym":set(SYNSETS_DEPTH_UP_TO_THIRD_LEVEL)})

DIR_LEXICAL_KNOWLEDGE = "/home/sakae/Windows/dataset/hypernym_detection/wordnet_most_frequent_sense/"
INCLUDE_INSTANCE_OF = False
if INCLUDE_INSTANCE_OF:
    suffix = "_incl_instance_of"
else:
    suffix = ""

# lexical knowledge: graded hyponymy relations extracted from WordNet corpus.
# lemmas in these files are case sensitive and phrase included.
cfg_hyponymy_relation_datasets = {
    "WordNet-hyponymy-noun": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, f"lexical_knowledge_wordnet_hyponymy_noun_valid_case_sensitive{suffix}.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: noun",
    },
    "WordNet-hyponymy-noun-excl-top-synsets": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, f"lexical_knowledge_wordnet_hyponymy_noun_valid_case_sensitive{suffix}.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter":_exclude_top_synsets,
        "description": "WordNet-hyponymy relation dataset: noun",
    },
    "WordNet-hyponymy-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, f"lexical_knowledge_wordnet_hyponymy_verb_valid_case_sensitive{suffix}.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: verb",
    },
    "WordNet-hyponymy-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, f"lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive{suffix}.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "description": "WordNet-hyponymy relation dataset: noun and verb",
    },
    "WordNet-hyponymy-noun-verb-excl-top-synsets": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, f"lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive{suffix}.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter":_exclude_top_synsets,
        "description": "WordNet-hyponymy relation dataset: noun and verb",
    }
}

