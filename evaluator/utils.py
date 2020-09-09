#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Callable
import numpy as np

def total_variation_distance(array_x: np.ndarray, array_y: np.ndarray, average=False):

    assert array_x.shape == array_y.shape, "shape mismatch detected."

    ret = np.sum(np.abs(array_x - array_y), axis=-1)/2
    if ret.ndim > 1 and average:
        ret = np.mean(ret)

    return ret

def _antisymmetricity_score(v_1: float, v_2: float, coef_lambda: float):
    """
    degree of the anti-symmetry between two values.
    z = exp(-\lambda*d(v_1,v_2))
    d(v_1,v_2) = |v_1 + v_2|
    :param v_1: a value
    :param v_2: another value
    :param lambda: exponential decay parameter.
    :return: z
    """
    ret = np.exp(-coef_lambda*np.abs(v_1+v_2))

    return ret

def hypernymy_propensity_score(s_w1w2: float, s_w2w1: float, coef_lambda: float = 1.0, directionality: bool = False):
    """
    degree of the hypernymy between two values.
    if you specify `directionality` as False, the score will be irrelevant to the order.

    if directionality=False: z = g(s_{w1,w2}, s_{w2,w1}; \alpha) max(s_w1w2, s_w2w1)
    else: z = g(s_{w1,w2}, s_{w2,w1}; \alpha) s_w1w2

    :param v_1: hypernymy_score s(w1, w2)
    :param v_2: hypernymy_score s(w2, w1)
    :param coef_lambda: exponential decay parameter of the antisymmetricity score function g().
    :return: z
    """
    g = _antisymmetricity_score(s_w1w2, s_w2w1, coef_lambda)
    if directionality:
        h = s_w1w2
    else:
        h = max(s_w1w2, s_w2w1)

    return g * h

#
def calc_signed_distance_from_hyperplane(mat_sample: np.ndarray, func_metric: Callable[[np.array], float], eps: float = 1E-10) -> np.array:
    """
    calculate the signed distance from the point to the hyperplane in a multi-dimensional space.
    let's say we specify `max` function as the func_metric and original space is K-dimensional (i.e. mat_sample.shape = (N,K))
    in this case, we assume the K-1 dimensional hyperplane that is supported by the points {max(mat_sample[:,0]), max(mat_sample[:,1]), ..., max(mat_sample[:,K-1])}.
    return will be the signed distance (right-hand direction is the positive value) from the hyperplane to the points.

    :param mat_sample: points to be evaluated. shape must be (N_sample, K_dim).
    :param func_metric: statistical metric that computes the representative point of each dimension. recommended metrics are; max, mean, median.

    :return: signed distance form the hyperplane to the points.
    """
    assert isinstance(mat_sample, np.ndarray), "`mat_sample` must be 2-D numpy array."
    assert hasattr(func_metric, "__call__"), "`func_metric` must be a callable object."
    N_K = mat_sample.shape[1]
    # single variable -> already you know the answer
    if N_K == 1:
        return mat_sample.ravel()
    # 1. calculate hyperplane
    vec_w_dash = np.apply_along_axis(func_metric, axis=0, arr=mat_sample)
    vec_w_dash_inv = np.divide(1., vec_w_dash, out=np.zeros_like(vec_w_dash), where=np.abs(vec_w_dash)>eps)
    vec_w = np.concatenate([vec_w_dash_inv, [-1.]]) # w = [{1/np.max(z_K)}, 1.0]
    vec_w /= (np.sqrt(np.sum(vec_w[:N_K]**2)) + eps) # L2 normalization: \sum_{k=1 to K}{w_k**2} == 1
    # 2. calculate integrated score
    vec_score = mat_sample.dot(vec_w[:N_K]) + vec_w[N_K]

    return vec_score
