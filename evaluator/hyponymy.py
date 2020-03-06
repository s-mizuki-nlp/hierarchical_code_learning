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
from sklearn.metrics import precision_recall_curve, roc_curve

from model.loss_supervised import HyponymyScoreLoss
from . import utils

array_like = Union[torch.Tensor, np.ndarray]

class SoftHyponymyPredictor(object):

    def __init__(self,
                 threshold_hyponymy_propensity_score: Optional[float] = 0.0,
                 threshold_soft_hyponymy_score: Optional[float] = 0.0):
        # this is used to compute soft code length
        self._auxiliary = HyponymyScoreLoss()
        self._threshold_hyponymy_propensity_score = threshold_hyponymy_propensity_score
        self._threshold_soft_hyponymy_score = threshold_soft_hyponymy_score

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

    def _calc_optimal_threshold_for_f_value(self, y_true, probas_pred, verbose: bool = True, **kwargs):

        def _f1_score_safe(prec, recall):
            if prec == recall == 0.0:
                return 0.0
            else:
                return 2*prec*recall/(prec+recall)

        # compute the threshold that maximizes f-value.
        v_prec, v_recall, v_threshold = precision_recall_curve(y_true=y_true, probas_pred=probas_pred, **kwargs)
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

        return threshold_opt

    def _calc_optimal_threshold_for_accuracy(self, y_true, probas_pred, verbose: bool = True, **kwargs):

        # compute the threshold that maximizes accuracy using receiver operating curve.
        v_tpr, v_fpr, v_threshold = roc_curve(y_true=y_true, y_score=probas_pred, **kwargs)
        n_sample = len(y_true)
        n_positive = np.sum(np.array(y_true ) == True)
        n_negative = n_sample - n_positive
        v_accuracy = (v_tpr*n_positive + (1-v_fpr)*n_negative)/n_sample

        idx = np.nanargmax(v_accuracy)
        threshold_opt = v_threshold[idx]

        if verbose:
            report = {
                "threshold_opt": threshold_opt,
                "tpr": v_tpr[idx],
                "fpr": v_fpr[idx],
                "accuracy": v_accuracy[idx]
            }
            pprint(report)

        return threshold_opt

    def calc_soft_hyponymy_score(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like):
        t_code_prob_x = self._numpy_to_tensor(mat_code_prob_x)
        t_code_prob_y = self._numpy_to_tensor(mat_code_prob_y)
        s_xy = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_x, t_code_prob_y).item()

        return s_xy

    def calc_hyponymy_propensity_score(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like, directionality: bool = False):
        s_xy = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)
        s_yx = self.calc_soft_hyponymy_score(mat_code_prob_y, mat_code_prob_x)

        # calculate hyponymy propensity score
        score = utils.hypernymy_propensity_score(s_xy, s_yx, directionality=directionality)

        return score

    def predict_directionality(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like) -> str:
        """
        assume (x,y) is hyponymy relation, it predicts which one, x or y, is hypernym

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the entity y
        :return: "x" if x is hypernym, "y" otherwise.
        """
        # x: hypernym, y: hyponym
        s_xy = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)
        # x: hyponym, y: hypernym
        s_yx = self.calc_soft_hyponymy_score(mat_code_prob_y, mat_code_prob_x)

        if s_xy > s_yx:
            return "x"
        else:
            return "y"

    def predict_is_hyponymy_relation(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like, threshold: Optional[float] = None) -> bool:
        """
        it predicts whether (x,y) pair is hyponymy or other relations (c.f. co-hyponymy, reverse-hyponymy, ...).
        this function is order-dependent. when you swap the order of arguments, response may be different.

        :param mat_code_prob_x: code probability of the hypernym candidate
        :param mat_code_prob_y: code probability of the hyponym candidate
        :param threshold: minimum value of the hyponymy propensity score to be classified as positive.
        """
        # calculate hyponymy propensity score
        score = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)

        # compare score with the threshold.
        threshold = self._threshold_soft_hyponymy_score if threshold is None else threshold

        # clasify if it is hyponymy or not
        ret = score > threshold

        return ret

    def predict_hyponymy_relation_by_hponymy_propensity_score(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like, threshold: Optional[float] = None):
        """
        it predicts what relation of the (x,y) pair holds among hyponymy, reverse-hyponymy, and other relations.
        this function is order-dependent only if (x,y) pair is either hyponymy or reverse-hyponymy relation.

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the other entity y
        :param threshold: minimum value of the hyponymy propensity score to be classified as positive.
        :return: "hyponymy", "reverse-hyponymy", or "other"
        """

        # calculate hyponymy propensity score
        score = self.calc_hyponymy_propensity_score(mat_code_prob_x, mat_code_prob_y, directionality=False)

        # compare score with the threshold.
        threshold = self._threshold_hyponymy_propensity_score if threshold is None else threshold

        # clasify if it is (reverse) hyponymy or not
        is_hyponymy = score > threshold

        # if false, classify as "other" relation.
        if not is_hyponymy:
            return "other"

        # if true, re-classify whether (forward) hyponymy or reverse hyponymy.
        direction = self.predict_directionality(mat_code_prob_x, mat_code_prob_y)
        if direction == "x":
            return "hyponymy"
        else:
            return "reverse-hyponymy"

    def predict_hyponymy_relation(self, mat_code_prob_x: array_like, mat_code_prob_y: array_like, threshold: Optional[float] = None) -> str:
        """
        it predicts what relation of the (x,y) pair holds among hyponymy, reverse-hyponymy, and other relations.
        this function is order-dependent only if (x,y) pair is either hyponymy or reverse-hyponymy relation.

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the other entity y
        :param threshold: minimum value of the hyponymy score to be classified as hyponymy relation
        :return: "hyponymy", "reverse-hyponymy", or "other"
        """

        # calculate hyponymy score for both forward and reversed direction
        score_forward = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)
        score_reversed = self.calc_soft_hyponymy_score(mat_code_prob_y, mat_code_prob_x)

        # compare score with the threshold.
        threshold = self._threshold_soft_hyponymy_score if threshold is None else threshold

        # if either one direction is true, relation must be either hyponymy or reverse hyponymy.
        if max(score_forward, score_reversed) > threshold:
            # re-classify whether (forward) hyponymy or reverse hyponymy.
            direction = self.predict_directionality(mat_code_prob_x, mat_code_prob_y)
            if direction == "x":
                return "hyponymy"
            else:
                return "reverse-hyponymy"
        # otherwise, relation must be "other".
        else:
            return "other"

    @property
    def thresholds(self):
        ret = {
            "hyponymy_propensity_score":self._threshold_hyponymy_propensity_score,
            "soft_hyponymy_score":self._threshold_soft_hyponymy_score
        }
        return ret

    @property
    def CLASS_LABELS_DIRECTIONALITY(self):
        return {"x","y"}

    @property
    def CLASS_LABELS_RELATION(self):
        return {"hyponymy","reverse-hyponymy","other"}