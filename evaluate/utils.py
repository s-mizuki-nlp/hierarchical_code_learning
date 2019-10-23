#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np

def total_variation_distance(array_x: np.ndarray, array_y: np.ndarray, average=False):

    assert array_x.shape == array_y.shape, "shape mismatch detected."

    ret = np.sum(np.abs(array_x - array_y), axis=-1)/2
    if ret.ndim > 1 and average:
        ret = np.mean(ret)

    return ret