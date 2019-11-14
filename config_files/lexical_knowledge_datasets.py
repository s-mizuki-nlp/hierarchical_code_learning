#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os

DIR_LEXICAL_KNOWLEDGE = "/home/sakae/Windows/dataset/hypernym_detection/wordnet_nguyen_2017/"

# lexical knowledge: graded hyponymy relations extracted from WordNet corpus.
# lemmas in these files are case sensitive and phrase included.
cfg_evaluation_datasets_ranking_retrieval = {
    "WordNet-hyponymy-noun": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos": 3},
        "description": "WordNet-hyponymy relation dataset: noun",
    },
    "WordNet-hyponymy-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos": 3},
        "description": "WordNet-hyponymy relation dataset: verb",
    },
    "WordNet-hyponymy-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lexical_knowledge_wordnet_hyponymy_noun_verb_valid_case_sensitive.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":0, "hypernym":1, "distance":2, "pos": 3},
        "description": "WordNet-hyponymy relation dataset: noun and verb",
    }
}
