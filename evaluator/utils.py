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

# def _antisymmetricity_score_old(v_1: float, v_2: float, lrelu_alpha: float):
#     """
#     degree of the anti-symmetry between two values.
#     z = \frac{1}{1+\alpha} LReLU(- \rho_2 / \rho_1; \alpha) + \frac{\alpha}{1+\alpha}
#     if |x| > |y|: \rho_1 = x, \rho_2 = y
#     else: \rho_1 = x, \rho_2 = y
#
#     :param v_1: a value
#     :param v_2: another value
#     :param lrelu_alpha: coefficient of the leaky relu. y=x if x>0 else alpha*x
#     :return: z
#     """
#     def lrelu(z, alpha):
#         return z if z > 0 else z*alpha
#
#     x, y = v_1, v_2
#     if np.abs(x) > np.abs(y):
#         rho1, rho2 = x, y
#     else:
#         rho1, rho2 = y, x
#     coef = 1 / (1+lrelu_alpha)
#     intercept = lrelu_alpha / (1+lrelu_alpha)
#     ret = coef * lrelu(-rho2/rho1, lrelu_alpha) + intercept
#
#     return ret

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