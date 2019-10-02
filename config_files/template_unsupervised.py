#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io

from model.loss import ReconstructionLoss, MutualInformationLoss
from model.discretizer import StraightThroughEstimator, GumbelSoftmax, Entmax15Estimator

hyper_parameters = {
    "model": {
        "encoder-decoder":{
            # "n_dim_emb": None, # determined by dataset
            "n_digits": 32,
            "n_ary": 32
        },
        "autoencoder":{
            "discretizer":Entmax15Estimator(add_gumbel_noise=True, temperature=1.0),
            "normalize_output_length": False
        }
    },
    "trainer": {
        "loss_reconst": ReconstructionLoss(),
        "loss_mutual_info": None,
        "learning_rate":0.001
    },
    "dataset": {
        "batch_size":128,
        "shuffle":True
    }
}

experiment_system = {
    "checkpoint_callback":{
        "filepath": None,
        "save_best_only": True,
        "verbose": False,
        "monitor": "val_loss"
    },
    "experiment":{
        "save_dir": None,
        "name": None
    },
    "max_nb_epochs": None,
    "progress_bar": True,
    "gpus": None
}