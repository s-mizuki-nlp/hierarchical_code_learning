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

def calc_lowest_common_ancestor_length(vec_x, vec_y):
    ret = 0
    for x, y in zip(vec_x, vec_y):
        if (x == y) and (x != 0) and (y != 0):
            ret += 1
        else:
            break
    return ret

def is_hypernymy_relation(vec_x, vec_y):
    for x, y in zip(vec_x, vec_y):
        if (x == 0) and (y != 0):
            return 1
        elif (x == y) and (x != 0) and (y != 0):
            continue
        else:
            return 0
    return 0

def calc_hyponymy_score(vec_x, vec_y):
    is_hypernym = is_hypernymy_relation(vec_x, vec_y)
    l_lca = calc_lowest_common_ancestor_length(vec_x, vec_y)
    l_hyper = calc_code_length(vec_x)
    l_hypo = calc_code_length(vec_y)

    score = is_hypernym*(l_hypo - l_hyper) + (1-is_hypernym)*(l_lca - l_hyper)
    return score


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
    ret = 1.0 - (np.sum(vec_prob_c_x*vec_prob_c_y) - vec_prob_c_x[0]*vec_prob_c_y[0])
    return ret

def calc_ancestor_probability(mat_prob_c_x, mat_prob_c_y):
    n_digits, n_ary = mat_prob_c_x.shape
    # vec_beta[n] = p_x[n][0]*(1-p_y[n][0])
    vec_beta = mat_prob_c_x[:,0]*(1-mat_prob_c_y[:,0])
    # vec_gamma[n] = \sum_{m=1 to M-1}p_x[n-1][m]*p_y[n-1][m]; vec_gamma[n] = 1.0
    vec_gamma_hat = np.sum(mat_prob_c_x[:,1:]*mat_prob_c_y[:,1:], axis=1)
    vec_gamma = np.concatenate(([1.0], vec_gamma_hat))[:n_digits]
    ret = np.sum(vec_beta*np.cumprod(vec_gamma))
    return ret

def calc_synonym_probability(mat_prob_c_x, mat_prob_c_y):
    # ret = \prod_{d}\sum_{a}p_x[d][a]*p_y[d][a]
    vec_gamma = np.sum(mat_prob_c_x*mat_prob_c_y, axis=-1)
    ret = np.prod(vec_gamma, axis=-1)
    return ret

def calc_soft_lowest_common_ancestor_length(mat_prob_c_x, mat_prob_c_y):
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
    vec_prob_c_y_zero = mat_prob_c_y[:,0]
    alpha = calc_ancestor_probability(mat_prob_c_x, mat_prob_c_y)
    beta = calc_synonym_probability(mat_prob_c_x, mat_prob_c_y)
    l_lca = calc_soft_lowest_common_ancestor_length(mat_prob_c_x, mat_prob_c_y)
    l_hyper = calc_soft_code_length(vec_prob_c_x_zero)
    l_hypo = calc_soft_code_length(vec_prob_c_y_zero)
    score = alpha*(l_hypo - l_hyper) + (1.-alpha-beta)*(l_lca - l_hyper)

    return score


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