#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys, io, os
from dataset import word_embeddings

DIR_WORD_EMBEDDINGS = "/home/sakae/Windows/public_model/embedding/"

cfg_word_embeddings = {
    "Word2Vec-google-news": {
        "path_word2vec_format":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-google-news-300/word2vec-google-news-300"),
        "binary":True,
        "init_sims":False,
        "transform":None,
        "enable_phrase_composition":True
    },
    "fastText-wiki-news": {
        "path_fasttext_binary_format":os.path.join(DIR_WORD_EMBEDDINGS, "fasttext-wiki-news-300/wiki-news-300d-1M-subword.bin"),
        "transform":None,
        "enable_phrase_composition":True
    },
    "Word2Vec-GloVe-wiki-gigaword": {
        "path_word2vec_format":os.path.join(DIR_WORD_EMBEDDINGS, "glove-wiki-gigaword-200/glove-wiki-gigaword-200.gz"),
        "binary":True,
        "init_sims":False,
        "transform":None,
        "enable_phrase_composition":True
    },
    "General-word2vec-umbc-sample": {
        "path_numpy_array_binary_format":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-umbc-corpus-sample/w2v_cased_200_win5_sgns1.npy"),
        "path_vocabulary_text":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-umbc-corpus-sample/w2v_cased_200_win5_sgns1.vocab"),
        "transform":None,
        "enable_phrase_composition":True
    },
    "Word2Vec-encow16ax-cased": {
        "path_word2vec_format":os.path.join(DIR_WORD_EMBEDDINGS, "word2vec-encow16ax-corpus/w2v_gensim_vector_cased_100_win5_min10_neg15_sgns1.bin"),
        "mmap":"r",
        "init_sims":False,
        "transform":None,
        "enable_phrase_composition":True
    }
}