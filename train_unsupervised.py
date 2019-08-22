#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional
import os, sys, io
from pprint import pprint
import argparse
import importlib.util
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from data.embeddings import ToyEmbeddingsDataset
from wrapper.trainer import UnsupervisedTrainer
from model.autoencoder import AutoEncoder
from model.encoder import SimpleEncoder
from model.decoder import SimpleDecoder
from model.loss import ReconstructionLoss
from data.embeddings import Word2VecDataset, FastTextDataset, Wikipedia2VecDataset

_RANDOM_SEED = 0

def parse_args():

    def check_ratio(value: str):
        ratio = float(value)
        assert 0.0 < ratio < 1.0, argparse.ArgumentTypeError(f"invalid value was specified: {value}")
        return ratio

    def check_gpus(value: Optional[str] = None):
        if value is None:
            return value
        gpus = [int(gpuid) for gpuid in value.split(",")]
        return gpus

    parser = argparse.ArgumentParser(description="train hierarchical code learning with unsupervised setting.")
    parser.add_argument("--embeddings", "-e", required=True, type=str, help="path to the embeddings object.")
    parser.add_argument("--embeddings_type", "-t", required=True, type=str, choices=("word2vec","fasttext","wikipedia2vec"),
                        help="type of the embeddings model.")
    parser.add_argument("--batch_size", "-b", required=False, type=int, default=128, help="minibatch size.")
    parser.add_argument("--epochs", required=False, type=int, default=10, help="maximum number of epochs.")
    parser.add_argument("--validation_split", "-s", required=False, type=check_ratio, default=0.0, help="validation split ratio. DEFAULT:0.0 (=disabled)")
    parser.add_argument("--config_file", "-c", required=True, type=str, help="path to the config file (*.py).")
    parser.add_argument("--log_dir", "-l", required=False, default="./log", type=str, help="directory to be used for saving tensorboard log.")
    parser.add_argument("--experiment_name", "-n", required=False, default="default", type=str, help="experiment name that is used to log results.")
    parser.add_argument("--saved_model_dir", "-m", required=False, default="./saved_model", type=str, help="directory to be used for saving trained model.")
    parser.add_argument("--gpus", required=False, type=check_gpus, default=None, help="GPU device ids to be used for training. DEFAULT:None(=cpu)"),
    parser.add_argument("--verbose", action="store_true", help="show verbose output.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    if args.verbose:
        print("\n=== arguments ===")
        cfg_args = {arg: getattr(args, arg) for arg in vars(args)}
        pprint(cfg_args)

    # load config file
    path_config = os.path.abspath(args.config_file)
    spec = importlib.util.spec_from_file_location('config', path_config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if args.verbose:
        print("\n=== configurations ===")
        pprint(config.hyper_parameters)

    # instanciate dataloader
    print(f"loading embeddings: {args.embeddings}")
    ## dataset
    if args.embeddings_type == "word2vec":
        dataset = Word2VecDataset(args.embeddings)
    elif args.embeddings_type == "fasttext":
        dataset = FastTextDataset(args.embeddings)
    elif args.embeddings_type == "wikipedia2vec":
        dataset = Wikipedia2VecDataset(args.embeddings)
    else:
        raise NotImplementedError(f"unsupported embeddings type: {args.embeddings_type}")

    ## dataloader
    n_sample = len(dataset)
    n_sample_val = int(n_sample*args.validation_split)
    if n_sample_val > 0:
        np.random.seed(_RANDOM_SEED)
        dataset_train, dataset_val = random_split(dataset, lengths=[n_sample - n_sample_val, n_sample_val])
    else:
        dataset_train = dataset
        dataset_val = None
    cfg_dataset = config.hyper_parameters["dataset"]
    cfg_dataset["batch_size"] = args.batch_size
    dataloader_train = DataLoader(dataset_train, **cfg_dataset)
    dataloader_val = None if dataset_val is None else DataLoader(dataset_val, **cfg_dataset)

    # instanciate autoencoder
    hyper_parameters = config.hyper_parameters
    cfg_model = config.hyper_parameters["model"]
    cfg_model["n_dim_emb"] = dataset.n_dim if cfg_model["n_dim_emb"] is None else cfg_model["n_dim_emb"]
    encoder = SimpleEncoder(**cfg_model)
    decoder = SimpleDecoder(**cfg_model)
    model = AutoEncoder(encoder=encoder, decoder=decoder, **cfg_model)

    # instanciate trainer
    cfg_trainer = config.hyper_parameters["trainer"]
    trainer = UnsupervisedTrainer(model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, **cfg_trainer)

    # instanciate experiment system
    cfg_system = config.experiment_system
    cfg_system["max_nb_epochs"] = args.epochs
    cfg_system["gpus"] = args.gpus
    cfg_system["checkpoint_callback"].filepath = args.saved_model_dir
    cfg_system["experiment"].save_dir = args.log_dir
    cfg_system["experiment"].name = args.experiment_name

    if args.verbose:
        cfg_system_print = {}
        for var, val in cfg_system.items():
            cfg_system_print[var] = str(val)
        print("=== experiment settings ===")
        pprint(cfg_system_print)

    system = Trainer(**cfg_system)
    system.fit(trainer)

    print("finished. good-bye.")