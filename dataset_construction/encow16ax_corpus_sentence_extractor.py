#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

## originally downloaded from below url.
# https://github.com/m-hahn/recursive-prd/tree/master/process_data

import sys, io
import html
import gzip
import progressbar
from pprint import pprint

_TOKEN_DELIMITER = "\t"
_EOS_TAG = "</s>"
_LOWERCASE = False

lst_input_path = sys.argv[1:-1]
path_output = sys.argv[-1]

"""
usage:
./encow16ax_corpus_sentence_extractor.py ./encow16ax*.xml.gz ./tokenized_corpus_case_sensitive.txt
"""

print(f"lowercase: {_LOWERCASE}")
print("inputs:")
pprint(lst_input_path)
print(f"output: {path_output}")

ofs = io.open(path_output, mode="w")
num_tokens = 0; num_sentences = 0
buff = []
q = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
q.update(0)
for path_input_b in lst_input_path:
    print(f"input file: {path_input_b}")
    ifs = gzip.open(path_input_b, 'r')
    for line in ifs:
        num_tokens += 1
        if num_tokens % 500000 == 0:
            q.update(num_tokens)
        try:
            line = line.decode().strip()
        except UnicodeDecodeError:
            print("[ERROR] Unicode Decode Error")
            print(line)
            line = line.decode("iso-8859-1").strip()
            print(line)
            continue
        if line == _EOS_TAG:
            sentence = " ".join(buff)
            if _LOWERCASE:
                sentence = sentence.lower()
            ofs.write(sentence + "\n")
            ofs.flush()
            buff = []
            num_sentences += 1
        if line.startswith("<") and line.endswith(">"):
            pass
        else:
            idx = line.find(_TOKEN_DELIMITER)
            if idx == -1:
                print("[ERROR] No Tab Found")
                print(line)
            else:
                token = line[:idx]
                token = html.unescape(token) if token.startswith("&") else token
                buff.append(token)

    ifs.close()
    print(f"so far: number of tokens: {num_tokens}, number of sentences: {num_sentences}")
del q

print(f"done.\nnumber of tokens: {num_tokens}, number of sentences: {num_sentences}")