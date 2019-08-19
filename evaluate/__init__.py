#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
import argparse

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

parser = argparse.ArgumentParser(description="hoge")
parser.add_argument("ARG_FULL", "ARG_SHORT", required=(True | False), [type = TYPE,] [default = None,] [
    choices = LIST,] help = "MESSAGE")
# type には __call__ method が使えるオブジェクトを指定する（例： float, int, unicode など）．
args = parser.parse_args()
