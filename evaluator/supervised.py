#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Callable, Iterable, Any, List, Union
from collections import defaultdict
import pydash
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dataset.word_embeddings import AbstractWordEmbeddingsDataset
from model.autoencoder import AutoEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from .hyponymy import HyponymyScoreBasedPredictor, EntailmentProbabilityBasedPredictor
from scipy.stats import spearmanr, kendalltau

class BaseEvaluator(object, metaclass=ABCMeta):

    def __init__(self, model: AutoEncoder,
                 hyponymy_predictor_type: str = "hyponymy_score",
                 embeddings_dataset: Optional[AbstractWordEmbeddingsDataset] = None,
                 evaluation_dataset: Optional[Dataset] = None,
                 **kwargs_dataloader):

        self._model = model
        if embeddings_dataset is not None:
            self._embeddings_dataset = embeddings_dataset
            self._embeddings_data_loader = DataLoader(embeddings_dataset, **kwargs_dataloader)
        if evaluation_dataset is not None:
            self._evaluation_dataset = evaluation_dataset
            self._evaluation_data_loader = DataLoader(evaluation_dataset, **kwargs_dataloader)

        if hyponymy_predictor_type == "hyponymy_score":
            self._hyponymy_predictor_class = HyponymyScoreBasedPredictor
        elif hyponymy_predictor_type == "entailment_probability":
            self._hyponymy_predictor_class = EntailmentProbabilityBasedPredictor
        else:
            valid_hyponymy_predictor_type = ("hyponymy_score", "entailment_probability")
            raise ValueError(f"`hyponymy_predictor_type` must be: {','.join(valid_hyponymy_predictor_type)}")

        self._default_evaluator = {
            "accuracy": lambda y_true, y_pred, **kwargs: accuracy_score(y_true, y_pred),
            "confusion_matrix": lambda y_true, y_pred, **kwargs: confusion_matrix(y_true, y_pred, **kwargs),
            "classification_report": lambda y_true, y_pred, **kwargs: classification_report(y_true, y_pred, **kwargs),
            "macro_f_value": lambda y_true, y_pred, **kwargs: f1_score(y_true, y_pred, average="macro", **kwargs),
        }

        self._update_task_specific_evaluator()

    def _tensor_to_list(self, tensor_or_list: Union[torch.Tensor, List]):
        if isinstance(tensor_or_list, torch.Tensor):
            return tensor_or_list.tolist()
        else:
            return tensor_or_list

    def _inference(self, dict_inference_functions: Dict[str, Callable[[np.array, np.array], Any]],
                   hyponym_field_name: str, hypernym_field_name: str, embedding_field_name: str):

        dict_lst_inference = defaultdict(list)
        for batch in self._evaluation_data_loader:
            # take hyponyms, hypernyms
            lst_hyponyms = batch[hyponym_field_name]
            lst_hypernyms = batch[hypernym_field_name]
            # take embeddings
            mat_emb_hyponyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hyponyms])
            mat_emb_hypernyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hypernyms])
            # encode embeddings into the code probabilities
            t_mat_code_prob_hyponyms = self._model.encode_soft(mat_emb_hyponyms)
            t_mat_code_prob_hypernyms = self._model.encode_soft(mat_emb_hypernyms)

            # apply predictor functions
            # x: hypernym, y: hyponym
            for mat_hyper, mat_hypo in zip(t_mat_code_prob_hypernyms, t_mat_code_prob_hyponyms):
                for inference_name, inference_function in dict_inference_functions.items():
                    ret = inference_function(mat_hyper, mat_hypo)
                    dict_lst_inference[inference_name].append(ret)

        return dict_lst_inference

    def _get_specific_field_values(self, target_field_name):
        lst_ret = []
        for batch in self._evaluation_data_loader:
            obj = batch[target_field_name]
            if torch.is_tensor(obj) or isinstance(obj, list):
                lst_ret.extend(self._tensor_to_list(obj))
            else:
                lst_ret.append(obj)
        return lst_ret

    @abstractmethod
    def _update_task_specific_evaluator(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class CodeLengthEvaluator(BaseEvaluator):

    """
    evaluator for code length prediction task
    """

    def _update_task_specific_evaluator(self):
        self._default_evaluator["scaled_mean_absolute_error"] = self._scaled_mean_absolute_error

    def _scaled_mean_absolute_error(self, y_true, y_pred, **kwargs):
        return np.mean(np.abs(y_pred/self._model.n_digits - y_true/np.max(y_true)))

    def evaluate(self, embedding_key_name: str = "embedding",
                    ground_truth_key_path: str = "entity_info.code_length",
                    evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                    **kwargs_for_metric_function):

        evaluator = self._default_evaluator if evaluator is None else evaluator

        lst_code_length = []
        lst_code_length_gt = []
        for batch in self._embeddings_data_loader:
            t_x = pydash.objects.get(batch, embedding_key_name)
            t_code_length_gt = pydash.objects.get(batch, ground_truth_key_path)

            t_code = self._model._encode(t_x)

            v_code_length = np.count_nonzero(t_code, axis=-1)
            v_code_length_gt = t_code_length_gt.data.numpy()

            lst_code_length.append(v_code_length)
            lst_code_length_gt.append(v_code_length_gt)

        v_code_length = np.concatenate(lst_code_length)
        v_code_length_gt = np.concatenate(lst_code_length_gt)

        dict_ret = {}

        for metric_name, f_metric in evaluator.items():
            try:
                dict_ret[metric_name] = f_metric(v_code_length_gt, v_code_length, **kwargs_for_metric_function)
            except:
                dict_ret[metric_name] = None

        return v_code_length_gt, v_code_length, dict_ret


class HyponymyDirectionalityEvaluator(BaseEvaluator):

    """
    evaluator for hypernymy directionality classification task
    """

    def _update_task_specific_evaluator(self):
        pass

    def evaluate(self, hyponym_field_name: str = "hyponym",
                hypernym_field_name: str = "hypernym",
                class_label_field_name: str = "class",
                embedding_field_name: str = "embedding",
                evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                **kwargs_for_metric_function):

        predictor = self._hyponymy_predictor_class()
        evaluator = self._default_evaluator if evaluator is None else evaluator

        # do prediction
        dict_inference_functions = {
            "predicted_class": lambda mat_hyper, mat_hypo: predictor.predict_directionality(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo)
        }
        dict_inference = self._inference(dict_inference_functions,
                                         hypernym_field_name=hypernym_field_name, hyponym_field_name=hyponym_field_name,
                                         embedding_field_name=embedding_field_name)
        lst_pred = dict_inference["predicted_class"]
        lst_gt = self._get_specific_field_values(target_field_name=class_label_field_name)

        # calculate metrics
        dict_ret = {}
        for metric_name, f_metric in evaluator.items():
            try:
                dict_ret[metric_name] = f_metric(lst_gt, lst_pred, **kwargs_for_metric_function)
            except:
                dict_ret[metric_name] = None

        return lst_gt, lst_pred, dict_ret


class BinaryHyponymyClassificationEvaluator(BaseEvaluator):

    """
    evaluator for binary hypernymy relation classification task
    """

    def _update_task_specific_evaluator(self):
        self._default_evaluator["accuracy_by_category"] = self._accuracy_by_category
        self._default_evaluator["area_under_curve"] = self._area_under_curve
        self._default_evaluator["optimal_threshold"] = self._optimal_threshold

    def _accuracy_by_category(self, lst_gt, lst_pred, lst_category, **kwargs):
        dict_denom = defaultdict(int)
        dict_num = defaultdict(int)
        for gt, pred, category in zip(lst_gt, lst_pred, lst_category):
            dict_denom[category] += 1
            if gt == pred:
                dict_num[category] += 1
        dict_acc = {}
        for category in dict_denom.keys():
            dict_acc[category] = dict_num[category] / dict_denom[category]

        return dict_acc

    def _area_under_curve(self, lst_gt, lst_score, **kwargs):
        ret = roc_auc_score(lst_gt, lst_score)
        return ret

    def _optimal_threshold(self, lst_gt, lst_score, **kwargs):
        predictor = self._hyponymy_predictor_class()
        ret = predictor.calc_optimal_threshold_accuracy(y_true=lst_gt, probas_pred=lst_score, verbose=True)
        return ret

    def evaluate(self, hyponym_field_name: str = "hyponym",
                hypernym_field_name: str = "hypernym",
                class_label_field_name: str = "class",
                embedding_field_name: str = "embedding",
                category_field_name: str = "relation",
                threshold_soft_hyponymy_score: float = 0.0,
                evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                **kwargs_for_metric_function):

        predictor = self._hyponymy_predictor_class(threshold=threshold_soft_hyponymy_score)
        evaluator = self._default_evaluator if evaluator is None else evaluator

        # do prediction
        dict_inference_functions = {
            "predicted_class": lambda mat_hyper, mat_hypo: predictor.predict_is_hyponymy_relation(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo),
            "predicted_score": lambda mat_hyper, mat_hypo: predictor.infer_score(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo)
        }
        dict_inference = self._inference(dict_inference_functions,
                                         hypernym_field_name=hypernym_field_name, hyponym_field_name=hyponym_field_name,
                                         embedding_field_name=embedding_field_name)
        lst_pred = dict_inference["predicted_class"]
        lst_score = dict_inference["predicted_score"]

        lst_gt = self._get_specific_field_values(target_field_name=class_label_field_name)
        lst_category = self._get_specific_field_values(target_field_name=category_field_name)

        # calculate metrics
        dict_ret = {}
        for metric_name, f_metric in evaluator.items():
            try:
                if metric_name == "accuracy_by_category":
                    dict_ret[metric_name] = f_metric(lst_gt, lst_pred, lst_category, **kwargs_for_metric_function)
                elif metric_name in ("area_under_curve", "optimal_threshold"):
                    dict_ret[metric_name] = f_metric(lst_gt, lst_score)
                else:
                    dict_ret[metric_name] = f_metric(lst_gt, lst_pred, **kwargs_for_metric_function)
            except:
                dict_ret[metric_name] = None

        return lst_gt, lst_pred, dict_ret


class MultiClassHyponymyClassificationEvaluator(BaseEvaluator):

    """
    evaluator for multi-class hyponymy relation classification task
    classes are: hyponymy, reverse-hyponymy, and other
    """

    def _update_task_specific_evaluator(self):
        self._default_evaluator["accuracy_by_category"] = self._accuracy_by_category
        self._default_evaluator["area_under_curve"] = self._area_under_curve
        self._default_evaluator["optimal_threshold"] = self._optimal_threshold

    def _accuracy_by_category(self, lst_gt, lst_pred, lst_category, **kwargs):
        dict_denom = defaultdict(int)
        dict_num = defaultdict(int)
        for gt, pred, category in zip(lst_gt, lst_pred, lst_category):
            dict_denom[category] += 1
            if gt == pred:
                dict_num[category] += 1
        dict_acc = {}
        for category in dict_denom.keys():
            dict_acc[category] = dict_num[category] / dict_denom[category]

        return dict_acc

    def _area_under_curve(self, lst_gt, lst_score, **kwargs):
        ret = roc_auc_score(lst_gt, lst_score)
        return ret

    def _optimal_threshold(self, lst_gt, lst_score, **kwargs):
        predictor = self._hyponymy_predictor_class()
        ret = predictor.calc_optimal_threshold_accuracy(y_true=lst_gt, probas_pred=lst_score, verbose=True)
        return ret

    def evaluate(self, hyponym_field_name: str = "hyponym",
                 hypernym_field_name: str = "hypernym",
                 class_label_field_name: str = "class",
                 embedding_field_name: str = "embedding",
                 category_field_name: str = "relation",
                 threshold_soft_hyponymy_score: float = 0.0,
                 evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                 **kwargs_for_metric_function):

        evaluator = self._default_evaluator if evaluator is None else evaluator
        predictor = self._hyponymy_predictor_class(threshold=threshold_soft_hyponymy_score)

        # do prediction
        dict_inference_functions = {
            "predicted_class": lambda mat_hyper, mat_hypo: predictor.predict_hyponymy_relation(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo),
            "predicted_score": lambda mat_hyper, mat_hypo: max(
                predictor.infer_score(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo),
                predictor.infer_score(mat_code_prob_x=mat_hypo, mat_code_prob_y=mat_hyper)
                )
        }
        dict_inference = self._inference(dict_inference_functions,
                                         hypernym_field_name=hypernym_field_name, hyponym_field_name=hyponym_field_name,
                                         embedding_field_name=embedding_field_name)
        lst_pred = dict_inference["predicted_class"]
        lst_score = dict_inference["predicted_score"]

        lst_gt = self._get_specific_field_values(target_field_name=class_label_field_name)
        lst_gt_binary = [gt in ("hyponymy", "reverse-hyponymy") for gt in lst_gt]
        lst_category = self._get_specific_field_values(target_field_name=category_field_name)

        # calculate metrics
        dict_ret = {}
        for metric_name, f_metric in evaluator.items():
            try:
                if metric_name == "accuracy_by_category":
                    dict_ret[metric_name] = f_metric(lst_gt, lst_pred, lst_category, **kwargs_for_metric_function)
                elif metric_name in ("area_under_curve", "optimal_threshold"):
                    dict_ret[metric_name] = f_metric(lst_gt_binary, lst_score)
                else:
                    dict_ret[metric_name] = f_metric(lst_gt, lst_pred, **kwargs_for_metric_function)
            except:
                dict_ret[metric_name] = None

        return lst_gt, lst_pred, dict_ret


class GradedLexicalEntailmentEvaluator(BaseEvaluator):

    """
    evaluator for graded lexical entailment task.
    this task requires the prediction of the degree of lexical entailment of a given word pair.
    """

    def _update_task_specific_evaluator(self):
        self._default_evaluator = {} # make it empty
        self._default_evaluator["spearman_rho"] = self._spearman_rho
        self._default_evaluator["kendall_tau"] = self._kendall_tau

    def _spearman_rho(self, lst_gt, lst_pred, **kwargs):
        rho, p_value = spearmanr(lst_gt, lst_pred)
        return rho

    def _kendall_tau(self, lst_gt, lst_pred, **kwargs):
        tau, p_value = kendalltau(lst_gt, lst_pred)
        return tau

    def evaluate(self, hyponym_field_name: str = "hyponym",
                 hypernym_field_name: str = "hypernym",
                 rating_field_name: str = "rating",
                 embedding_field_name: str = "embedding",
                 evaluator: Optional[Dict[str, Callable[[Iterable, Iterable],Any]]] = None,
                 **kwargs_for_metric_function):

        evaluator = self._default_evaluator if evaluator is None else evaluator
        predictor = self._hyponymy_predictor_class()

        # do prediction
        dict_inference_functions = {
            "predicted_score": lambda mat_hyper, mat_hypo: predictor.infer_score(mat_code_prob_x=mat_hyper, mat_code_prob_y=mat_hypo)
        }
        dict_inference = self._inference(dict_inference_functions,
                                         hypernym_field_name=hypernym_field_name, hyponym_field_name=hyponym_field_name,
                                         embedding_field_name=embedding_field_name)
        lst_pred = dict_inference["predicted_score"]

        # get ground-truth rating
        lst_gt = self._get_specific_field_values(target_field_name=rating_field_name)
        # get ground-truth tuple: (hypernym, hyponym, rating)
        g = map(self._get_specific_field_values, (hypernym_field_name, hyponym_field_name, rating_field_name))
        lst_tup_gt = list(zip(*g))

        # calculate metrics
        dict_ret = {}
        for metric_name, f_metric in evaluator.items():
            try:
                dict_ret[metric_name] = f_metric(lst_gt, lst_pred, **kwargs_for_metric_function)
            except:
                dict_ret[metric_name] = None

        return lst_tup_gt, lst_pred, dict_ret
