#!/usr/bin/env python
# -*- coding:utf-8 -*-

import unittest
import numpy as np
import torch
from . import utils_bugfix
from model.loss_supervised import HyponymyScoreLoss


class HyponymyScoreLossLayer(unittest.TestCase):

    _EPS = 1E-5

    def setUp(self) -> None:

        n_ary = 6
        n_digits = 4
        tau = 4.0

        self._n_ary = n_ary
        self._n_digits = n_digits

        vec_p_x_zero = np.array([0.02,0.05,0.8,0.8])
        vec_x_repr = np.array([1,2,0,0])
        vec_p_y_zero = np.array([0.02,0.05,0.1,0.6])
        vec_y_repr = np.array([1,2,3,0])
        mat_p_x_1 = utils_bugfix.generate_probability_matrix(vec_p_x_zero, vec_x_repr, n_digits, n_ary, tau)
        mat_p_y_1 = utils_bugfix.generate_probability_matrix(vec_p_y_zero, vec_y_repr, n_digits, n_ary, tau)

        vec_p_x_zero = np.array([0.02,0.05,0.05,0.8])
        vec_x_repr = np.array([1,2,3,0])
        vec_p_y_zero = np.array([0.02,0.05,0.6,0.6])
        vec_y_repr = np.array([1,2,0,0])
        mat_p_x_2 = utils_bugfix.generate_probability_matrix(vec_p_x_zero, vec_x_repr, n_digits, n_ary, tau)
        mat_p_y_2 = utils_bugfix.generate_probability_matrix(vec_p_y_zero, vec_y_repr, n_digits, n_ary, tau)

        # x:hypernym, y:hyponym
        # mat_p_*: (n_dim, n_ary)
        self._mat_p_x = mat_p_x_1
        self._mat_p_y = mat_p_y_1
        # arry_p_*: (n_batch, n_dim, n_ary)
        self._arry_p_x = np.stack([mat_p_x_1, mat_p_x_2])
        self._arry_p_y = np.stack([mat_p_y_1, mat_p_y_2])
        self._arry_p_batch = np.stack([mat_p_x_1, mat_p_x_2, mat_p_y_1, mat_p_y_2])
        # train_signal: (hypernym_index, hyponym_index, hyponymy_score)
        self._hyponymy_tuples = [(0, 2, 1.0), (1, 3, -1.0)] # [(x1, y1, 1.0), (x2, y2, -1.0)]

        self._t_mat_p_x = torch.from_numpy(self._mat_p_x)
        self._t_mat_p_y = torch.from_numpy(self._mat_p_y)
        self._t_arry_p_x = torch.from_numpy(self._arry_p_x)
        self._t_arry_p_y = torch.from_numpy(self._arry_p_y)
        self._t_arry_p_batch = torch.from_numpy(self._arry_p_batch)

        self._normalize_code_length = True
        self._normalize_coefficient_for_ground_truth = 1/3.0
        self._loss_layer = HyponymyScoreLoss(normalize_hyponymy_score=self._normalize_code_length,
                                             normalize_coefficient_for_ground_truth=self._normalize_coefficient_for_ground_truth,
                                             distance_metric="mse")

    def test_intensity_to_probability(self):

        t_test = self._t_mat_p_x[:,0]
        arry_test = t_test.data.numpy()

        expected = utils_bugfix._intensity_to_probability(arry_test)
        actual = self._loss_layer._intensity_to_probability(t_test).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_intensity_to_probability_two_dim(self):

        t_test = self._t_arry_p_x[:,:,0]
        arry_test = t_test.data.numpy()

        expected = np.stack(list(map(utils_bugfix._intensity_to_probability, arry_test)))
        actual = self._loss_layer._intensity_to_probability(t_test).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_soft_code_length(self):

        t_test = self._t_arry_p_x
        arry_test = t_test.data.numpy()

        expected = np.array([utils_bugfix.calc_soft_code_length(mat_p[:,0]) for mat_p in arry_test])
        actual = self._loss_layer.calc_soft_code_length(t_test).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_break_intensity(self):

        t_test_x = self._t_mat_p_x
        t_test_y = self._t_mat_p_y
        arry_test_x = t_test_x.data.numpy()
        arry_test_y = t_test_y.data.numpy()

        expected = np.array([utils_bugfix._calc_break_intensity(v_x, v_y) for v_x, v_y in zip(arry_test_x, arry_test_y)])
        actual = self._loss_layer._calc_break_intensity(t_test_x, t_test_y).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_break_intensity_two_dim(self):

        def calc_break_intensity_(mat_x, mat_y):
            return np.array([utils_bugfix._calc_break_intensity(v_x, v_y) for v_x, v_y in zip(mat_x, mat_y)])

        t_test_x = self._t_arry_p_x
        t_test_y = self._t_arry_p_y
        arry_test_x = t_test_x.data.numpy()
        arry_test_y = t_test_y.data.numpy()

        expected = np.stack([calc_break_intensity_(mat_x, mat_y) for mat_x, mat_y in zip(arry_test_x, arry_test_y)])
        actual = self._loss_layer._calc_break_intensity(t_test_x, t_test_y).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_soft_least_common_ancestor_length(self):

        t_test_x = self._t_arry_p_x
        t_test_y = self._t_arry_p_y
        arry_test_x = t_test_x.data.numpy()
        arry_test_y = t_test_y.data.numpy()

        expected = np.array([utils_bugfix.calc_soft_least_common_ancestor_length(mat_x, mat_y) for mat_x, mat_y in zip(arry_test_x, arry_test_y)])
        actual = self._loss_layer.calc_soft_least_common_ancestor_length(t_test_x, t_test_y).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_soft_hyponymy_score(self):

        t_test_x = self._t_arry_p_x
        t_test_y = self._t_arry_p_y
        arry_test_x = t_test_x.data.numpy()
        arry_test_y = t_test_y.data.numpy()

        expected = np.array([utils_bugfix.calc_soft_hyponymy_score(mat_x, mat_y) for mat_x, mat_y in zip(arry_test_x, arry_test_y)])
        actual = self._loss_layer.calc_soft_hyponymy_score(t_test_x, t_test_y).data.numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_loss_value(self):

        t_test = self._t_arry_p_batch
        lst_train = self._hyponymy_tuples
        arry_test = t_test.data.numpy()

        lst_idx_x = [tup[0] for tup in lst_train]
        lst_idx_y = [tup[1] for tup in lst_train]
        y_true = np.array([tup[2] for tup in lst_train])
        arry_test_x = arry_test[lst_idx_x]
        arry_test_y = arry_test[lst_idx_y]

        y_pred = np.array([utils_bugfix.calc_soft_hyponymy_score(mat_x, mat_y) for mat_x, mat_y in zip(arry_test_x, arry_test_y)])

        if self._normalize_code_length:
            y_pred /= self._n_digits
            y_true *= self._normalize_coefficient_for_ground_truth
        expected = np.mean((y_pred - y_true)**2) # L2 loss
        actual = self._loss_layer.forward(t_test, lst_train)

        self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
