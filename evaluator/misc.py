#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
from .hyponymy import SoftHyponymyPredictor
from .supervised import BaseEvaluator

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
                 normalize: bool = True):

        predictor = SoftHyponymyPredictor()
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
