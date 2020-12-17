#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional, Dict
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_curve

def convert_class_to_label(ground_truth_class: str) -> int:
    map_class_to_label = {
        "hyponymy":1,
        "hyper":1,
        "reverse-hyponymy":-1,
        "rhyper":-1,
        "other":0
    }
    return map_class_to_label[ground_truth_class]


# source: https://github.com/facebookresearch/hypernymysuite/blob/master/hypernymysuite/evaluation.py
def wbless_setup(ground_truth_labels: np.ndarray, predicted_score: np.ndarray, is_in_vocab: np.ndarray,
                 optional_attributes: Optional[Dict[str, np.ndarray]] = {},
                 validation_ratio: Optional[float] = 0.02, n_trials: Optional[int] = 1000, random_seed: Optional[int] = 42,
                 **kwargs):
    """
    Accuracy using a threshold, with a dataset that explicitly contains reverse pairs.

    @param ground_truth_labels: array of binary labels. 1:hyponymy, 0:other
    @param predicted_score: predicted hyponymy propensity score. larger tend to be hyponymy.
    @param is_in_vocab: array of True/False flags. True:in-vocbulary, False:out-of-vocabulary.
    @param optional_attributes: dict of the array of optional attributes (i.e. ground-truth label). Array size must be identical to the sample size.
    @param validation_ratio: ratio of the validation data. DEFAULT:0.02
    @param n_trials: number of iteration of evaluation. DEFAULT:1000
    @return: average of metrics; validation accuracy, test accuracy, and optimal threshold.
    """

    # Ensure we always get the same results
    rng = np.random.RandomState(random_seed)
    VAL_PROB = validation_ratio
    NUM_TRIALS = n_trials

    # We have no way of handling oov
    h = predicted_score[is_in_vocab]
    y = ground_truth_labels[is_in_vocab]

    n_sample = len(y)
    n_val_sample = int(np.ceil(n_sample*VAL_PROB))

    val_scores = []
    test_scores = []
    thresholds = []
    dict_optional_test_scores = defaultdict(list)

    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        ## m_val = rng.rand(len(y)) < VAL_PROB

        m_val = np.repeat(False, n_sample)
        val_args = rng.choice(n_sample, size=n_val_sample, replace=False)
        m_val[val_args] = True

        assert len(m_val) == n_sample
        assert sum(m_val) == n_val_sample

        # Test is everything except val
        m_test = ~m_val
        _, _, t = precision_recall_curve(y[m_val], h[m_val])
        # pick the highest accuracy on the validation set
        thr_accs = np.mean((h[m_val, np.newaxis] >= t) == y[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]
        preds_val = h[m_val] >= best_t
        preds_test = h[m_test] >= best_t
        # Evaluate
        vec_is_val_correct = preds_val == y[m_val]
        vec_is_test_correct = preds_test == y[m_test]
        val_scores.append(np.mean(vec_is_val_correct))
        test_scores.append(np.mean(vec_is_test_correct))
        thresholds.append(best_t)
        # sanity check
        assert np.allclose(val_scores[-1], thr_accs.max())

        # test accuracy for optional attributes (i.e. ground-truth label)
        for attr_name, vec_attributes in optional_attributes.items():
            for attr_value in np.unique(vec_attributes):
                metric_name = f"cv_test_accuracy_{attr_name}-{attr_value}"
                mask_test_attr_v = vec_attributes[m_test] == attr_value
                if sum(mask_test_attr_v) > 0:
                    test_score_attr_v = np.mean(vec_is_test_correct[mask_test_attr_v])
                else:
                    test_score_attr_v = None
                dict_optional_test_scores[metric_name].append(test_score_attr_v)

    # report average across many folds
    dict_results = {
        "cv_validation_accuracy": np.mean(val_scores),
        "cv_test_accuracy": np.mean(test_scores),
        "cv_optimal_threshold": np.mean(thresholds)
    }
    # report optional accuracy across meny folds
    if len(optional_attributes) > 0:
        dict_optional_results = {metric_name:np.nanmean(lst_test_scores) for metric_name, lst_test_scores in dict_optional_test_scores.items()}
        dict_results.update(dict_optional_results)

    return dict_results


# source: https://github.com/facebookresearch/hypernymysuite/blob/master/hypernymysuite/evaluation.py
def bibless_setup(ground_truth_labels: np.ndarray,
                  predicted_score_forward: np.ndarray, predicted_score_reverse: np.ndarray,
                  is_in_vocab: np.ndarray,
                  optional_attributes: Optional[Dict[str, np.ndarray]] = {},
                  validation_ratio: Optional[float] = 0.02, n_trials: Optional[int] = 1000, random_seed: Optional[int] = 42,
                  **kwargs) -> Dict[str, float]:
    """
    Accuracy using a threshold, with a dataset that explicitly contains reverse pairs.

    @param ground_truth_labels: array of binary labels. 1:hyponymy, -1:reverse-hyponymy, 0:other
    @param predicted_score_forward: predicted hyponymy propensity score of (hyponymy, hypernymy) tuple.
    @param predicted_score_reverse: predicted hyponymy propensity score of (hypernymy, hyponymy) tuple.
    @param is_in_vocab: array of True/False flags. True:in-vocbulary, False:out-of-vocabulary.
    @param optional_attributes: dict of the array of optional attributes (i.e. ground-truth labels). Array size must be identical to the sample size.
    @param validation_ratio: ratio of the validation data. DEFAULT:0.02
    @param n_trials: number of iteration of evaluation. DEFAULT:1000
    @return: average of metrics; validation accuracy, test accuracy, and optimal threshold.
    """

    rng = np.random.RandomState(random_seed)
    VAL_PROB = validation_ratio
    NUM_TRIALS = n_trials

    # We have no way of handling oov
    y = ground_truth_labels[is_in_vocab]

    # hypernymy could be either direction
    y_binary = y != 0

    # get forward and backward predictions
    h_forward = predicted_score_forward[is_in_vocab]
    h_reverse = predicted_score_reverse[is_in_vocab]
    h_binary = np.max([h_forward, h_reverse], axis=0)

    dir_pred = 2 * np.float32(h_forward >= h_reverse) - 1

    n_sample = len(y)
    n_val_sample = int(np.ceil(n_sample*VAL_PROB))

    val_scores = []
    test_scores = []
    thresholds = []
    dict_optional_test_scores = defaultdict(list)

    for _ in range(NUM_TRIALS):
        # Generate a new mask every time
        ## m_val = rng.rand(len(y)) < VAL_PROB
        m_val = np.repeat(False, n_sample)
        val_args = rng.choice(n_sample, size=n_val_sample, replace=False)
        m_val[val_args] = True

        assert len(m_val) == n_sample
        assert sum(m_val) == n_val_sample

        # Test is everything except val
        m_test = ~m_val

        # set the threshold based on the maximum score
        _, _, t = precision_recall_curve(y_binary[m_val], h_binary[m_val])
        thr_accs = np.mean((h_binary[m_val, np.newaxis] >= t) == y_binary[m_val, np.newaxis], axis=0)
        best_t = t[thr_accs.argmax()]

        det_preds_val = h_binary[m_val] >= best_t
        det_preds_test = h_binary[m_test] >= best_t

        fin_preds_val = det_preds_val * dir_pred[m_val]
        fin_preds_test = det_preds_test * dir_pred[m_test]

        # evaluate
        vec_is_val_correct = fin_preds_val == y[m_val]
        vec_is_test_correct = fin_preds_test == y[m_test]

        val_scores.append(np.mean(vec_is_val_correct))
        test_scores.append(np.mean(vec_is_test_correct))
        thresholds.append(best_t)

        # test accuracy for optional attributes (i.e. ground-truth label)
        for attr_name, vec_attributes in optional_attributes.items():
            for attr_value in np.unique(vec_attributes):
                metric_name = f"cv_test_accuracy_{attr_name}-{attr_value}"
                mask_test_attr_v = vec_attributes[m_test] == attr_value
                if sum(mask_test_attr_v) > 0:
                    test_score_attr_v = np.mean(vec_is_test_correct[mask_test_attr_v])
                else:
                    test_score_attr_v = None
                dict_optional_test_scores[metric_name].append(test_score_attr_v)

    # report average across many folds
    dict_results = {
        "cv_validation_accuracy": np.mean(val_scores),
        "cv_test_accuracy": np.mean(test_scores),
        "cv_optimal_threshold": np.mean(thresholds)
    }
    # report optional accuracy across meny folds
    if len(optional_attributes) > 0:
        dict_optional_results = {metric_name:np.nanmean(lst_test_scores) for metric_name, lst_test_scores in dict_optional_test_scores.items()}
        dict_results.update(dict_optional_results)

    return dict_results
