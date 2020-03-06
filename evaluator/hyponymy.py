#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from typing import Union, Optional, Tuple, Iterable
from pprint import pprint
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from model.loss_supervised import HyponymyScoreLoss
from . import utils

Array_like = Union[torch.Tensor, np.ndarray]

class BasePredictor(object, metaclass=ABCMeta):

    def __init__(self,
                 threshold: Optional[float] = 0.0):
        # this is used to compute soft code length
        self._auxiliary = HyponymyScoreLoss()
        self._threshold = threshold

    def _tensor_to_numpy(self, object: Array_like) -> np.ndarray:
        if isinstance(object, torch.Tensor):
            return object.cpu().numpy()
        elif isinstance(object, np.ndarray):
            return object
        else:
            raise TypeError(f"unsupported type: {type(object)}")

    def _numpy_to_tensor(self, object: Array_like) -> torch.Tensor:
        if isinstance(object, torch.Tensor):
            return object
        elif isinstance(object, np.ndarray):
            return torch.from_numpy(object)
        else:
            raise TypeError(f"unsupported type: {type(object)}")

    def calc_optimal_threshold_fvalue(self, y_true, probas_pred, verbose: bool = True, **kwargs):

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

    def calc_optimal_threshold_accuracy(self, y_true, probas_pred, verbose: bool = True, **kwargs):

        # compute the threshold that maximizes accuracy using receiver operating curve.
        v_fpr, v_tpr, v_threshold = roc_curve(y_true=y_true, y_score=probas_pred, **kwargs)
        n_sample = len(y_true)
        n_positive = np.sum(np.array(y_true) == True)
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

    def calc_soft_hyponymy_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like):
        t_code_prob_x = self._numpy_to_tensor(mat_code_prob_x)
        t_code_prob_y = self._numpy_to_tensor(mat_code_prob_y)
        s_xy = self._auxiliary.calc_soft_hyponymy_score(t_code_prob_x, t_code_prob_y).item()

        return s_xy

    def calc_hyponymy_propensity_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, directionality: bool = False):
        s_xy = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)
        s_yx = self.calc_soft_hyponymy_score(mat_code_prob_y, mat_code_prob_x)

        # calculate hyponymy propensity score
        score = utils.hypernymy_propensity_score(s_xy, s_yx, directionality=directionality)

        return score

    def calc_entailment_probability(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like):
        t_code_prob_x = self._numpy_to_tensor(mat_code_prob_x)
        t_code_prob_y = self._numpy_to_tensor(mat_code_prob_y)
        p_xy = self._auxiliary.calc_ancestor_probability(t_code_prob_x, t_code_prob_y).item()
        return p_xy

    @abstractmethod
    def THRESHOLD(self):
        pass

    @abstractmethod
    def predict_directionality(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like) -> str:
        """
        assume (x,y) is hyponymy relation, it predicts which one, x or y, is hypernym

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the entity y
        :return: "x" if x is hypernym, "y" otherwise.
        """

        pass

    @abstractmethod
    def predict_is_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> bool:
        """
        it predicts whether (x,y) pair is hyponymy or other relations (c.f. co-hyponymy, reverse-hyponymy, ...).
        this function is order-dependent. when you swap the order of arguments, response may be different.

        :param mat_code_prob_x: code probability of the hypernym candidate
        :param mat_code_prob_y: code probability of the hyponym candidate
        """

        pass

    @abstractmethod
    def predict_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> str:
        """
        it predicts what relation of the (x,y) pair holds among hyponymy, reverse-hyponymy, and other relations.
        this function is order-dependent only if (x,y) pair is either hyponymy or reverse-hyponymy relation.

        :param mat_code_prob_x: code probability of the entity x
        :param mat_code_prob_y: code probability of the other entity y
        :return: "hyponymy", "reverse-hyponymy", or "other"
        """

        pass

    @abstractmethod
    def infer_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like) -> str:
        """
        it returns the inferred score that is used for classification.
        conceptually, it represents a degree of hyponymy relation.

        @param mat_code_prob_x: code probability of the entity x
        @param mat_code_prob_y: code probability of the other entity y
        @return: scalar value
        """

        pass

    @property
    def CLASS_LABELS(self):
        return {
            "directionality":{"x","y"},
            "is_hyponymy_relation":{True, False},
            "hyponymy_relation":{"hyponymy","reverse-hyponymy","other"}
        }


class HyponymyScoreBasedPredictor(BasePredictor):

    def predict_directionality(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like) -> str:
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

    def predict_is_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> bool:

        # calculate hyponymy propensity score
        score = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)

        # compare score with the threshold.
        threshold = self._threshold if threshold is None else threshold

        # clasify if it is hyponymy or not
        ret = score > threshold

        return ret

    def predict_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> str:

        # calculate hyponymy score for both forward and reversed direction
        score_forward = self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)
        score_reversed = self.calc_soft_hyponymy_score(mat_code_prob_y, mat_code_prob_x)

        # compare score with the threshold.
        threshold = self._threshold if threshold is None else threshold

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

    def infer_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like):
        return self.calc_soft_hyponymy_score(mat_code_prob_x, mat_code_prob_y)

    @property
    def THRESHOLD(self):
        ret = {
            "hyponymy_score":self._threshold
        }
        return ret


class EntailmentProbabilityBasedPredictor(BasePredictor):

    def __init__(self,
                 threshold: Optional[float] = 0.5):
        """
        predicts hyponymy relation based on the entailment probability of entity pair.

        @param threshold: threshold of the entailment probability. larget than specified value is regarded as hyponymy relation.
        """
        self._auxiliary = HyponymyScoreLoss()
        self._threshold = threshold

    def predict_directionality(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like) -> str:
        # x: hypernym, y: hyponym
        p_xy = self.calc_entailment_probability(mat_code_prob_x, mat_code_prob_y)
        # x: hyponym, y: hypernym
        p_yx = self.calc_entailment_probability(mat_code_prob_y, mat_code_prob_x)

        if p_xy > p_yx:
            return "x"
        else:
            return "y"

    def predict_is_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> bool:

        # calculate entailment probability
        prob = self.calc_entailment_probability(mat_code_prob_x, mat_code_prob_y)

        # compare score with the threshold.
        threshold = self._threshold if threshold is None else threshold

        # clasify if it is hyponymy or not
        ret = prob > threshold

        return ret

    def predict_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None) -> str:

        # calculate hyponymy score for both forward and reversed direction
        p_forward = self.calc_entailment_probability(mat_code_prob_x, mat_code_prob_y)
        p_reversed = self.calc_entailment_probability(mat_code_prob_y, mat_code_prob_x)

        # compare score with the threshold.
        threshold = self._threshold if threshold is None else threshold

        # if either one direction is true, relation must be either hyponymy or reverse hyponymy.
        if max(p_forward, p_reversed) > threshold:
            # re-classify whether (forward) hyponymy or reverse hyponymy.
            direction = self.predict_directionality(mat_code_prob_x, mat_code_prob_y)
            if direction == "x":
                return "hyponymy"
            else:
                return "reverse-hyponymy"
        # otherwise, relation must be "other".
        else:
            return "other"

    def infer_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like) -> str:
        return self.calc_entailment_probability(mat_code_prob_x, mat_code_prob_y)

    @property
    def THRESHOLD(self):
        ret = {
            "inclusion_probability":self._threshold
        }
        return ret


class HyponymyPropensityScoreBasedPredictor(HyponymyScoreBasedPredictor):

    def predict_hyponymy_relation(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like, threshold: Optional[float] = None):

        # calculate hyponymy propensity score
        score = self.calc_hyponymy_propensity_score(mat_code_prob_x, mat_code_prob_y, directionality=False)

        # compare score with the threshold.
        threshold = self._threshold if threshold is None else threshold

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

    def infer_score(self, mat_code_prob_x: Array_like, mat_code_prob_y: Array_like):
        return self.calc_hyponymy_propensity_score(mat_code_prob_x, mat_code_prob_y, directionality=False)

    @property
    def THRESHOLD(self):
        ret = {
            "hyponymy_propensity_score":self._threshold
        }
        return ret
