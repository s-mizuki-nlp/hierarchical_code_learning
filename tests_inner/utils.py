#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np

def calc_code_length(vec_c):
    ret = 0
    for c in vec_c:
        if c == 0:
            break
        ret += 1
    return ret

def calc_common_prefix_length(vec_x, vec_y):
    # regard zero value in hypernym code as the wildcard
    code_length_y = calc_code_length(vec_y)
    ret = 0
    for x, y in zip(vec_x[:code_length_y], vec_y[:code_length_y]):
        if x == 0:
            ret += 1
        else:
            if x == y:
                ret += 1
            else:
                break
    return ret

def calc_hyponymy_score(vec_x, vec_y):
    cpl = calc_common_prefix_length(vec_x, vec_y)
    hcl = calc_code_length(vec_x)
    return cpl - hcl


def _intensity_to_probability(vec_intensity):
    vec_i = vec_intensity
    vec_prob = np.cumprod(1.0 - np.concatenate(([0.0], vec_i))) * np.concatenate((vec_i, [1.0]))
    return vec_prob

def calc_soft_code_length(vec_prob_c_zero):
    vec_p = vec_prob_c_zero
    n_digits = vec_p.size
    vec_p_at_n = _intensity_to_probability(vec_p)
    ret = np.sum(vec_p_at_n * np.arange(n_digits+1))

    return ret

def _calc_break_intensity(vec_prob_c_x, vec_prob_c_y):
    # x: hypernym, y: hyponym
    ret = 1.0 - vec_prob_c_x[0] - np.sum(vec_prob_c_x*vec_prob_c_y) + 2*vec_prob_c_x[0]*vec_prob_c_y[0]
    return ret

def calc_soft_common_prefix_length(mat_prob_c_x, mat_prob_c_y):
    n_digits, n_ary = mat_prob_c_x.shape
    vec_intensity = np.zeros(n_digits, dtype=np.float32)
    for idx, (vec_prob_c_x_t, vec_prob_c_y_t) in enumerate(zip(mat_prob_c_x, mat_prob_c_y)):
        vec_intensity[idx] = _calc_break_intensity(vec_prob_c_x_t, vec_prob_c_y_t)

    vec_prob_break = _intensity_to_probability(vec_intensity)
    ret = np.sum(vec_prob_break * np.arange(n_digits+1))

    return ret

def calc_soft_hyponymy_score(mat_prob_c_x, mat_prob_c_y):
    # calculate soft hyponymy score
    # prob_c[n,v] = Pr{C_n=v}; prob_c = (N_digits, N_ary)
    vec_prob_c_x_zero = mat_prob_c_x[:,0]
    hcl = calc_soft_code_length(vec_prob_c_x_zero)
    cpl = calc_soft_common_prefix_length(mat_prob_c_x, mat_prob_c_y)

    return cpl - hcl


def _softmax(vec_z):
    return np.exp(vec_z) / np.sum(np.exp(vec_z))

def generate_probability_matrix(vec_p_zero, vec_repr, n_digits, n_ary, tau):
    # generate toy probability matrix according to the specified probability and code representation.
    mat_p = np.zeros(shape=(n_digits, n_ary), dtype=np.float32)

    for idx in range(n_digits):
        vec_z = np.random.uniform(size=n_ary) + np.eye(n_ary)[vec_repr[idx]]*tau
        vec_p = _softmax(vec_z)
        p_zero_t = vec_p_zero[idx]
        vec_p_norm = np.concatenate(([p_zero_t], (1.0-p_zero_t)*vec_p[1:]/np.sum(vec_p[1:])))
        mat_p[idx,:] = vec_p_norm

    return mat_p

def sample_discrete_code(mat_prob, size):
    # generate probabilistic samples(=discrete codes) from specified probability distribution
    n_digits, n_ary = mat_prob.shape
    ret = np.zeros(shape=(size, n_digits), dtype=np.int)

    bins = np.arange(n_ary)
    for b in range(size):
        ret_b = np.zeros(n_digits, dtype=np.int)
        for n in range(n_digits):
            ret_b[n] = np.random.choice(bins, p=mat_prob[n])
            if ret_b[n] == 0:
                break
        ret[b] = ret_b

    return ret