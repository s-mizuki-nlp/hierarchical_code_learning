#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
from nltk.corpus import wordnet as wn

def synset_to_name(synset):
    return synset.name()

def get_hypernyms_and_distance_via_shortest_path(synset):
    dict_ret = {}
    for hypernym, distance in synset.hypernym_distances():
        if hypernym == synset:
            continue
        current_value = dict_ret.get(hypernym, float("inf"))
        if distance < current_value:
            dict_ret[hypernym] = distance

    return dict_ret

def get_synset_lemmas(synset):
    return synset.lemma_names()

def get_synset_lexeme(synset):
    return synset.name().split(".n")[0]

def _list_up_depth_to_root(synset, root_synset):
    ret = []
    for s, d in synset.hypernym_distances():
        if s == root_synset:
            ret.append(d)
    return ret

def get_synset_depth(synset, root_synset=wn.synset("entity.n.01")):
    # returns max, min, avg depth
    lst_depth = _list_up_depth_to_root(synset, root_synset)
    if len(lst_depth) == 0:
        return (None, None, None)
    else:
        return (float(np.max(lst_depth)), float(np.min(lst_depth)), float(np.mean(lst_depth)))