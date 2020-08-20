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

_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})
_exclude_synonymy = DictionaryFilter(excludes={"distance":{0}})
_exclude_synonymy_and_antonymy = DictionaryFilter(excludes={"distance":{0,-1}})

DIR_LEXICAL_KNOWLEDGE = "/home/sakae/Windows/dataset/hypernym_detection/vulic_2018/"

# lexical knowledge: graded hyponymy relations extracted from WordNet corpus.
# lemmas in these files are case sensitive and phrase included.
cfg_hyponymy_relation_datasets = {
    "hyponymy-synonymy-antonymy": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_hyponymy_synonymy_antonymy_valid.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": None,
        "description": "[Vulic and Mrksic, 2018] relation: hyponymy,synonymy,antonymy. pos:noun,verb.",
    },
    "hyponymy": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_hyponymy_synonymy_antonymy_valid.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _exclude_synonymy_and_antonymy,
        "description": "[Vulic and Mrksic, 2018] relation: hyponymy. pos:noun,verb.",
    },
    "hyponymy-antonymy": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_hyponymy_synonymy_antonymy_valid.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos":3, "synset_hyponym":4, "synset_hypernym":5},
        "transform": _distance_str_to_float,
        "filter": _exclude_synonymy,
        "description": "[Vulic and Mrksic, 2018] relation: hyponymy,antonymy. pos:noun,verb.",
    }
}

