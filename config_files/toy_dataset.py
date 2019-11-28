#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
from dataset.transform import FieldTypeConverter
_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})

def _relation_to_direction_class(sample):
    class_mapper = {"hyponymy":"x", "reverse-hyponymy":"y"}
    sample["class"] = class_mapper[sample["relation"]]
    return sample

def _relation_to_binary_class(sample):
    class_mapper = {"hyponymy":True}
    sample["class"] = class_mapper.get(sample["relation"], False)
    return sample

def _relation_to_three_classes(sample):
    class_mapper = {"hyponymy":"hyponymy", "reverse-hyponymy":"reverse-hyponymy"}
    sample["class"] = class_mapper.get(sample["relation"], "other")
    return sample


DIR_TOY_DATASET = "/home/sakae/Windows/dataset/hypernym_detection/toy_dataset_supervised_final"

cfg_hyponymy_relation_datasets = {
    "hyponymy-triple-train": {
        "path": os.path.join(DIR_TOY_DATASET, "hyponymy_triple_train.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "distance":2},
        "transform": _distance_str_to_float,
        "description": "toy dataset (trianset). n_sample=19344, distance=[1,2,3]"
    },
    "hyponymy-triple-dev": {
        "path": os.path.join(DIR_TOY_DATASET, "hyponymy_triple_dev.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "distance":2},
        "transform": _distance_str_to_float,
        "description": "toy dataset (development set). n_sample=2418, distance=[1,2,3]"
    },
    "hyponymy-triple-test": {
        "path": os.path.join(DIR_TOY_DATASET, "hyponymy_triple_test.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "distance":2},
        "transform": _distance_str_to_float,
        "description": "toy dataset (test set). n_sample=2419, distance=[1,2,3]"
    }
}

cfg_embedding_dataset = {
    "entity-and-embeddings": {
        "path_numpy_array_binary_format":os.path.join(DIR_TOY_DATASET, "embeddings.npy"),
        "path_vocabulary_text":os.path.join(DIR_TOY_DATASET, "vocabulary.txt"),
        "path_vocabulary_information_json":os.path.join(DIR_TOY_DATASET, "entities.json"),
        "transform":None,
        "enable_phrase_composition":False
    }
}

cfg_evaluation_datasets_classification = {
    "directionality": {
        "path": os.path.join(DIR_TOY_DATASET, "evalset_hyponymy.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "is_hyponymy":4, "relation": 2},
        "transform":_relation_to_direction_class,
        "description": "toy evalset: directionality - which is the hypernym"
    },
    "binary-hyponymy": {
        "path": os.path.join(DIR_TOY_DATASET, "evalset.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "is_hyponymy":4, "relation": 2},
        "transform":_relation_to_binary_class,
        "description": "toy evalset: binary clasification - hyponymy or other relations",
    },
    "multiclass-hyponymy": {
        "path": os.path.join(DIR_TOY_DATASET, "evalset.txt"),
        "header": True,
        "delimiter": "\t",
        "columns": {"hyponym":1, "hypernym":0, "is_hyponymy":4, "relation": 2},
        "transform":_relation_to_three_classes,
        "description": "toy evalset: three-class classification - hyponymy, reverse-hyponymy, or other relations",
    }
}
