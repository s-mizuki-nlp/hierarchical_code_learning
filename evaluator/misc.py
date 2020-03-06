#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional
import numpy as np
import pydash
from tests_inner.utils import calc_hyponymy_score
from dataset.taxonomy import WordNetTaxonomy
from .hyponymy import HyponymyScoreBasedPredictor
from .supervised import BaseEvaluator
from dataset.word_embeddings import AbstractWordEmbeddingsDataset
from model.autoencoder import AutoEncoder
from torch.utils.data import DataLoader, Dataset


class HyponymyScoreDistributionEvaluator(BaseEvaluator):

    """
    computes hyponymy scores of forward and backward direction.
    @return lists of s_ij, s_ji, category
    """

    def _update_task_specific_evaluator(self):
        pass

    def evaluate(self, hyponym_field_name: str = "hyponym",
                 hypernym_field_name: str = "hypernym",
                 embedding_field_name: str = "embedding",
                 category_field_name: str = "relation",
                 ground_truth: bool = False,
                 normalize: bool = True):

        predictor = HyponymyScoreBasedPredictor()
        n_digits = self._model.n_digits

        lst_s_ij = []; lst_s_ji = []; lst_category = []
        for batch in self._evaluation_data_loader:
            # take hyponyms, hypernyms
            lst_hyponyms = batch[hyponym_field_name]
            lst_hypernyms = batch[hypernym_field_name]
            # take embeddings
            mat_emb_hyponyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hyponyms])
            mat_emb_hypernyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hypernyms])
            # encode embeddings into the code probabilities
            _, t_mat_code_prob_hyponyms, _ = self._model.predict(mat_emb_hyponyms)
            _, t_mat_code_prob_hypernyms, _ = self._model.predict(mat_emb_hypernyms)

            # evaluate hyponymy score for both directions
            # suppose x is hypernym and y is hyponym
            for mat_x, mat_y in zip(t_mat_code_prob_hypernyms, t_mat_code_prob_hyponyms):
                score_ij = predictor.calc_soft_hyponymy_score(mat_code_prob_x=mat_x, mat_code_prob_y=mat_y)
                score_ji = predictor.calc_soft_hyponymy_score(mat_code_prob_x=mat_y, mat_code_prob_y=mat_x)
                if normalize:
                    score_ij /= n_digits
                    score_ji /= n_digits
                lst_s_ij.append(score_ij)
                lst_s_ji.append(score_ji)

            # store category information
            lst_category.extend(self._tensor_to_list(batch[category_field_name]))

        return lst_s_ij, lst_s_ji, lst_category


class HyponymyScoreDiffEvaluator(BaseEvaluator):

    """
    computes difference between predicted hyponymy score and ground-truth hyponymy score.
    @return lists of s_pred, s_gt, s_diff, category
    """

    def _update_task_specific_evaluator(self):
        pass

    def evaluate(self, hyponym_field_name: str = "hyponym",
                 hypernym_field_name: str = "hypernym",
                 embedding_field_name: str = "embedding",
                 category_field_name: str = "relation",
                 code_representation_key_path: Optional[str] = "entity_info.code_representation",
                 normalize: bool = True):

        predictor = HyponymyScoreBasedPredictor()
        n_digits = self._model.n_digits
        entity_to_code = lambda entity: pydash.objects.get(self._embeddings_dataset[entity], code_representation_key_path)

        n_code_length = 1
        lst_s_pred = []; lst_s_gt = []; lst_category = []
        for batch in self._evaluation_data_loader:
            # take hyponyms, hypernyms
            lst_hyponyms = batch[hyponym_field_name]
            lst_hypernyms = batch[hypernym_field_name]
            # take embeddings
            mat_emb_hyponyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hyponyms])
            mat_emb_hypernyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hypernyms])
            # encode embeddings into the code probabilities
            _, t_mat_code_prob_hyponyms, _ = self._model.predict(mat_emb_hyponyms)
            _, t_mat_code_prob_hypernyms, _ = self._model.predict(mat_emb_hypernyms)

            # take ground-truth code representations

            # evaluate predicted hyponymy score
            # suppose x is hypernym and y is hyponym
            for mat_x, mat_y in zip(t_mat_code_prob_hypernyms, t_mat_code_prob_hyponyms):
                score_pred = predictor.calc_soft_hyponymy_score(mat_code_prob_x=mat_x, mat_code_prob_y=mat_y)
                lst_s_pred.append(score_pred)

            # evaluate ground-truth hyponymy score
            lst_code_repr_hyponyms = map(entity_to_code, lst_hyponyms)
            lst_code_repr_hypernyms = map(entity_to_code, lst_hypernyms)
            for code_hypernym, code_hyponym in zip(lst_code_repr_hypernyms, lst_code_repr_hyponyms):
                score_gt = calc_hyponymy_score(code_hypernym, code_hyponym)
                n_code_length = max(n_code_length, len(code_hypernym), len(code_hyponym))
                lst_s_gt.append(score_gt)

            # store category information
            lst_category.extend(self._tensor_to_list(batch[category_field_name]))

        if normalize:
            lst_s_pred = [score / n_digits for score in lst_s_pred]
            lst_s_gt = [score / n_code_length for score in lst_s_gt]

        return lst_s_pred, lst_s_gt, lst_category



class WordNetHyponymyScoreDiffEvaluator(BaseEvaluator):

    """
    computes difference between predicted hyponymy score and hyponymy score induced from the trainset taxonomy.
    @return lists of s_pred, s_gt, s_diff, category
    """

    def __init__(self, model: AutoEncoder,
                 taxonomy: WordNetTaxonomy,
                 embeddings_dataset: Optional[AbstractWordEmbeddingsDataset] = None,
                 evaluation_dataset: Optional[Dataset] = None,
                 **kwargs_dataloader):

        super().__init__(model, embeddings_dataset, evaluation_dataset, **kwargs_dataloader)
        self._taxonomy = taxonomy

    def _update_task_specific_evaluator(self):
        pass

    def evaluate(self, hyponym_field_name: str = "hyponym",
                 hypernym_field_name: str = "hypernym",
                 embedding_field_name: str = "embedding",
                 category_field_name: str = "relation",
                 remove_null_score: bool = True,
                 normalize: bool = True):

        predictor = HyponymyScoreBasedPredictor()
        n_digits = self._model.n_digits

        n_code_length = 1
        lst_s_pred = []; lst_s_gt = []; lst_category = []
        for batch in self._evaluation_data_loader:
            # take hyponyms, hypernyms
            lst_hyponyms = batch[hyponym_field_name]
            lst_hypernyms = batch[hypernym_field_name]
            # take embeddings
            mat_emb_hyponyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hyponyms])
            mat_emb_hypernyms = np.stack([self._embeddings_dataset[entity][embedding_field_name] for entity in lst_hypernyms])
            # encode embeddings into the code probabilities
            _, t_mat_code_prob_hyponyms, _ = self._model.predict(mat_emb_hyponyms)
            _, t_mat_code_prob_hypernyms, _ = self._model.predict(mat_emb_hypernyms)

            # evaluate predicted hyponymy score
            # suppose x is hypernym and y is hyponym
            for mat_x, mat_y in zip(t_mat_code_prob_hypernyms, t_mat_code_prob_hyponyms):
                score_pred = predictor.calc_soft_hyponymy_score(mat_code_prob_x=mat_x, mat_code_prob_y=mat_y)
                lst_s_pred.append(score_pred)

            # evaluate ground-truth hyponymy score
            for hyponym, hypernym in zip(lst_hyponyms, lst_hypernyms):
                score_gt = self._taxonomy.hyponymy_score(hypernym=hypernym, hyponym=hyponym)
                lst_s_gt.append(score_gt)

            # store category information
            lst_category.extend(self._tensor_to_list(batch[category_field_name]))

        if remove_null_score:
            lst_s_pred_r = []; lst_s_gt_r = []; lst_category_r = []
            for pred, gt, category in zip(lst_s_pred, lst_s_gt, lst_category):
                if gt is None:
                    continue
                lst_s_pred_r.append(pred)
                lst_s_gt_r.append(gt)
                lst_category_r.append(category)
            lst_s_pred, lst_s_gt, lst_category = lst_s_pred_r, lst_s_gt_r, lst_category_r

        if normalize:
            lst_s_pred = [score / n_digits for score in lst_s_pred]
            lst_s_gt = [score / n_code_length for score in lst_s_gt]

        return lst_s_pred, lst_s_gt, lst_category
