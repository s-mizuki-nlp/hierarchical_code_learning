#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Union, Optional, Tuple, Iterable
from pprint import pprint
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

from model.loss import HyponymyScoreLoss
from . import utils

array_like = Union[torch.Tensor, np.ndarray]

class SoftHyponymyPredictor(object):

    def __init__(self, threshold_hyponymy_propensity_score: Optional[float] = 0.0):
        # this is used to compute soft code length
        self._auxiliary = HyponymyScoreLoss()
        self._threshold_hyponymy = threshold_hyponymy_propensity_score

    def optimize_hyponymy_propensity_score_threshold(self, development_set: Iterable[Tuple[array_like, array_like, bool]],
                                                   verbose: bool = True, **kwargs) -> None:
        """
        automatically adjust the threshold of the hyponymy propensity score so as to maximize the f-value on the development set.

        :param development_set: iterable of the tuple of (code probability of x, code probability of y, hyponymy or not)
        :param kwargs: keyword arguments passed to sklearn.metrics.precision_recall_curve() method.
        """

        lst_y_true = []
        lst_score = []

        for t_x, t_y, y_true in development_set:
            score = self._calc_hyponymy_propensity_score(t_x, t_y, directionality=False)
            lst_y_true.append(y_true)
            lst_score.append(score)

        # if all examples are positive, we have to nothing but take the minimum value.
        if all(lst_y_true):
            return min(lst_score)

        def _f1_score_safe(prec, recall):
            if prec == recall == 0.0:
                return 0.0
            else:
                return 2*prec*recall/(prec+recall)

        # otherwise, we compute the threshold that maximizes f-value.
        v_prec, v_recall, v_threshold = precision_recall_curve(y_true=lst_y_true, probas_pred=lst_score, **kwargs)
        v_f1_score = np.vectorize(_f1_score_safe)(v_prec, v_recall)
        idx = np.nanargmax(v_f1_score)
        threshold_opt = v_threshold[idx]

        if verbose:
            report = {
                "threshold_opt": threshold_opt,
                "precision": v_prec[idx],
                "recall": v_recall[idx],
                "f1-score": v_f1_score[idx]
            }
            pprint(report)

        print(f"threshold was set to: {threshold_opt}")
        self._threshold_hyponymy = threshold_opt

    def _tensor_to_numpy(self, object: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(object, torch.Tensor):
            return object.cpu().numpy()
        elif isinstance(object, np.ndarray):
            return object
        else:
            raise TypeError(f"unsupported type: {type(object)}")

    def _numpy_to_tensor(self, object: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(object, torch.Tensor):
            return object
        elif isinstance(object, np.ndarray):
            return torch.from_numpy(object)
        else:
            raise TypeError(f"unsupported type: {type(object)}")

    def _calc_hyponymy_propensity_score(self, mat_code_prob_x, mat_code_prob_y, directionality: bool = False):
        t_code_prob_x = self._numpy_to_tensor(mat_code_prob_x)
        t_code_prob_y = self._numpy_to_tensor(mat_code_prob_y)

        # assume x is hypernym and y is hyponym
        # ToDo: we may need to deal with the batch dimension.
        s_xy = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_x, t_code_prob_y).item()
        s_yx = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_y, t_code_prob_x).item()

        # calculate hyponymy propensity score
        score = utils.hypernymy_propensity_score(s_xy, s_yx, directionality=directionality)

        return score

    def predict_directionality(self, mat_code_prob_x, mat_code_prob_y):
        """
        assume (x,y) is hyponymy relation, it predicts which one, x or y, is hypernym

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the entity y
        :return: "x" if x is hypernym, "y" otherwise.
        """
        t_code_prob_x = self._numpy_to_tensor(mat_code_prob_x)
        t_code_prob_y = self._numpy_to_tensor(mat_code_prob_y)

        # x: hypernym, y: hyponym
        s_xy = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_x, t_code_prob_y).item()
        # x: hyponym, y: hypernym
        s_yx = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_y, t_code_prob_x).item()

        if s_xy > s_yx:
            return "x"
        else:
            return "y"

    def predict_hyponymy(self, mat_code_prob_x, mat_code_prob_y, threshold: Optional[float] = None):
        """
        it predicts whether (x,y) pair is either hyponymy or reverse-hyponymy relation, or other relations (c.f. co-hyponymy).
        this function is order agnostic. when you swap the order of arguments, response will be identical.

        :param mat_code_prob_x: code probability of the entity
        :param mat_code_prob_y: code probability of the other entity
        :param threshold: minimum value of the hyponymy propensity score to be classified as positive.
        """
        # calculate hyponymy propensity score
        score = self._calc_hyponymy_propensity_score(mat_code_prob_x, mat_code_prob_y, directionality=False)

        # compare score with the threshold.
        threshold = self._threshold_hyponymy if threshold is None else threshold

        # clasify if it is (reverse) hyponymy or not
        ret = score > threshold

        return ret

    def predict_relation(self, mat_code_prob_x, mat_code_prob_y, threshold: Optional[float] = None):
        """
        it predicts what relation of the (x,y) pair holds among hyponymy, reverse-hyponymy, and other relations.
        this function is order-dependent only if (x,y) pair is either hyponymy or reverse-hyponymy relation.

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the other entity y
        :param threshold: minimum value of the hyponymy propensity score to be classified as positive.
        :return: "hyponymy", "reverse-hyponymy", or "other"
        """

        # classify if (x,y) pair is (reverse) hyponymy or not.
        is_hyponymy = self.predict_hyponymy(mat_code_prob_x, mat_code_prob_y, threshold)

        # if false, classify as "other" relation.
        if not is_hyponymy:
            return "other"

        # if true, re-classify whether (forward) hyponymy or reverse hyponymy.
        direction = self.predict_directionality(mat_code_prob_x, mat_code_prob_y)
        if direction == "x":
            return "hyponymy"
        else:
            return "reverse-hyponymy"


