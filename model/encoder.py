#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import warnings
from typing import Optional, Dict, Any, Union
import copy
import inspect
import torch
from torch import nn
from torch.nn import functional as F
from .regressor import SoftmaxBasedCDFEstimator, ScheduledSoftmaxBasedCDFEstimator
from .functional import CodeProbabiltyAdjuster
from .encoder_internal import MultiDenseLayer, StackedLSTMLayer
from .loss_unsupervised import CodeValueMutualInformationLoss


class SimpleEncoder(nn.Module):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 dtype=torch.float32, **kwargs):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype

        self._build()

    def _build(self):

        self.x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)
        self.lst_h_to_z = nn.ModuleList([nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary) for n in range(self._n_digits)])

    def forward(self, input_x: torch.Tensor):

        t_h = torch.tanh(self.x_to_h(input_x))
        lst_z = [torch.log(F.softplus(h_to_z(t_h))) for h_to_z in self.lst_h_to_z]
        lst_prob_c = [F.softmax(t_z, dim=1) for t_z in lst_z]

        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def calc_code_probability(self, input_x: torch.Tensor, **kwargs):
        return self.forward(input_x)

    @property
    def use_built_in_discretizer(self):
        return False

    def _adjust_code_probability_to_monotone_increasing(self, probs: torch.Tensor, probs_prev: Union[None, torch.Tensor]):
        # if Pr{C_{d-1}} is not given, we don't adjust the probability.
        if probs_prev is None:
            return probs

        dtype, device = self._dtype_and_device(probs)
        n_ary = self._n_ary

        # adjust Pr{Cd=0} using stick-breaking process
        # probs_zero_*: (n_batch, 1)
        probs_zero = torch.index_select(probs, dim=-1, index=torch.tensor(0, device=device))
        probs_zero_prev = torch.index_select(probs_prev, dim=-1, index=torch.tensor(0, device=device))
        probs_zero_adj = probs_zero_prev + (1.0 - probs_zero_prev)*probs_zero

        # probs_nonzero_*: (n_batch, n_ary-1)
        probs_nonzero = torch.index_select(probs, dim=-1, index=torch.arange(1, n_ary, dtype=torch.long, device=device))
        adjust_factor = (1.0 - probs_zero_adj) / ((1.0 - probs_zero) + 1E-6)
        probs_nonzero_adj = adjust_factor * probs_nonzero

        # concatenate adjusted probabilities
        probs_adj = torch.cat((probs_zero_adj, probs_nonzero_adj), dim=-1)

        return probs_adj


class CodeLengthAwareEncoder(SimpleEncoder):

    _kwargs_stacked_lstm_layer = {
        "time_distributed":True,
        "n_layer":1,
        "bidirectional":False
    }
    _kwargs_multi_dense_layer = {
        "n_layer":3,
        "activation_function":F.relu,
    }
    _kwargs_code_length_predictor = {
        "n_mlp_layer":3,
        "assign_nonzero_value_on_most_significant_digit":False,
        "init_code_length":None
    }

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 internal_layer_class: Optional[nn.Module] = None,
                 code_length_predictor_class: Optional[nn.Module] = None,
                 dtype=torch.float32,
                 kwargs_code_length_predictor: Optional[Dict[str, Any]] = None,
                 kwargs_stacked_lstm_layer: Optional[Dict[str, Any]] = None,
                 kwargs_multi_dense_layer: Optional[Dict[str, Any]] = None,
                 **kwargs):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype
        self._code_length_predictor_class = code_length_predictor_class
        self._internal_layer_class = internal_layer_class
        self._internal_layer_class_type = None
        self._n_ary_internal = None

        self._build(kwargs_code_length_predictor, kwargs_stacked_lstm_layer, kwargs_multi_dense_layer)

    def _build(self, kwargs_code_length_predictor_: Optional[Dict[str, Any]] = None,
                kwargs_stacked_lstm_layer_: Optional[Dict[str, Any]] = None,
                kwargs_multi_dense_layer_: Optional[Dict[str, Any]] = None):

        # update default parameters
        kwargs_multi_dense_layer = copy.deepcopy(self._kwargs_multi_dense_layer)
        if isinstance(kwargs_multi_dense_layer_, dict):
            kwargs_multi_dense_layer.update(kwargs_multi_dense_layer_)

        kwargs_stacked_lstm_layer = copy.deepcopy(self._kwargs_stacked_lstm_layer)
        if isinstance(kwargs_stacked_lstm_layer_, dict):
            kwargs_stacked_lstm_layer.update(kwargs_stacked_lstm_layer_)

        kwargs_code_length_predictor = copy.deepcopy(self._kwargs_code_length_predictor)
        if isinstance(kwargs_code_length_predictor_, dict):
            kwargs_code_length_predictor.update(kwargs_code_length_predictor_)

        # x -> h: h = tanh(W*x+b)
        self.x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)

        # if adjuster class is specified, p(c_n) -> adjust so that p(c_n-1=0) <= p(c_n=0)
        # if Estimator class is specified, x -> p(c_n=0): p(c_n=0) = \sum_{d=1 to n} g_d(x)
        if (self._code_length_predictor_class is None) or \
            (CodeProbabiltyAdjuster in inspect.getmro(self._code_length_predictor_class)):
            self.code_length_predictor = CodeProbabiltyAdjuster.apply
            self._n_ary_internal = self._n_ary
        elif ScheduledSoftmaxBasedCDFEstimator in inspect.getmro(self._code_length_predictor_class):
            self.code_length_predictor = ScheduledSoftmaxBasedCDFEstimator(n_dim_input=self._n_dim_emb, n_output=self._n_digits,
                                                                           dtype=self._dtype,
                                                                           **kwargs_code_length_predictor)
            self._n_ary_internal = self._n_ary - 1
        elif SoftmaxBasedCDFEstimator in inspect.getmro(self._code_length_predictor_class):
            self.code_length_predictor = SoftmaxBasedCDFEstimator(n_dim_input=self._n_dim_emb, n_output=self._n_digits,
                                                                  dtype=self._dtype,
                                                                  **kwargs_code_length_predictor)
            self._n_ary_internal = self._n_ary - 1
        else:
            raise NotImplementedError(f"unsupported layer was specified: {self._code_length_predictor_class.__class__}")

        # h -> z_n: z_n = softplus(f_n(h))
        n_dim_h, n_dim_z = self._n_dim_hidden, self._n_ary_internal
        if (self._internal_layer_class is None) or (nn.Linear in inspect.getmro(self._internal_layer_class)):
            lst_layers = [nn.Linear(in_features=n_dim_h, out_features=n_dim_z) for _ in range(self._n_digits)]

            self._internal_layer_class_type = "Linear"

        elif MultiDenseLayer in inspect.getmro(self._internal_layer_class):
            lst_layers = []
            for _ in range(self._n_digits):
                l = MultiDenseLayer(n_dim_in=n_dim_h, n_dim_out=n_dim_z, n_dim_hidden=n_dim_h, bias=False, **kwargs_multi_dense_layer)
                lst_layers.append(l)

            self._internal_layer_class_type = "MultiDenseLayer"

        elif StackedLSTMLayer in inspect.getmro(self._internal_layer_class):
            l = StackedLSTMLayer(n_dim_in=n_dim_h, n_dim_out=n_dim_z, n_dim_hidden=n_dim_h, n_seq_len=self._n_digits,
                                 **kwargs_stacked_lstm_layer)
            if self._n_ary_internal == self._n_ary:
                init_code_length = kwargs_code_length_predictor.get("init_code_length", None)
                if init_code_length is None:
                    pass
                elif init_code_length == "min":
                    l.init_bias_to_min()
                elif init_code_length == "max":
                    l.init_bias_to_max()
            lst_layers = [l]

            self._internal_layer_class_type = "StackedLSTMLayer"

        else:
            raise NotImplementedError(f"unsupported layer was specified: {self._internal_layer_class.__class__}")

        self.lst_h_to_z = nn.ModuleList(lst_layers)

    def _calc_code_probability_by_stacked_lstm(self, input_x: torch.Tensor):
        stacked_lstm_layer = self.lst_h_to_z[0]

        # t_h: (n_batch, n_dim_internal)
        t_h = torch.tanh(self.x_to_h(input_x))
        # t_z: (n_batch, n_digits, n_ary_internal)
        t_z = torch.log(F.softplus(stacked_lstm_layer(t_h)) + 1E-6)
        # lst_z: [(n_batch, n_ary_internal)]*n_digits
        lst_z = list(map(torch.squeeze, t_z.split(1, dim=1)))
        # t_prob_c: (n_batch, n_digits, n_ary_internal)
        lst_prob_c = [F.softmax(t_z, dim=-1) for t_z in lst_z]
        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def _calc_code_probability_by_mlp(self, input_x: torch.Tensor):
        lst_mlp_layer = self.lst_h_to_z
        # t_h: (n_batch, n_dim_internal)
        t_h = torch.tanh(self.x_to_h(input_x))
        lst_z = [torch.log(F.softplus(mlp_layer_n(t_h)) + 1E-6) for mlp_layer_n in lst_mlp_layer]
        # t_prob_c: (n_batch, n_digits, n_ary_internal)
        lst_prob_c = [F.softmax(t_z, dim=-1) for t_z in lst_z]
        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def forward(self, input_x: torch.Tensor):

        # code probability: p(c_n=m)
        if self._internal_layer_class_type in ("Linear","MultiDenseLayer"):
            t_prob_c = self._calc_code_probability_by_mlp(input_x)
        elif self._internal_layer_class_type == "StackedLSTMLayer":
            t_prob_c = self._calc_code_probability_by_stacked_lstm(input_x)
        else:
            raise ValueError(f"unknown internal layer type: {self._internal_layer_class_type}")

        if self._n_ary_internal == self._n_ary - 1:
            # predict zero probability: p(c_n=0) separately using code length predictor
            # t_prob_c_zero: (n_batch, n_digits) -> (n_batch, n_digits, 1)
            t_prob_c_zero = self.code_length_predictor.forward(input_x)
            t_prob_c_zero = t_prob_c_zero.unsqueeze(-1)

            # concat it with code probability
            # t_prob_c: (n_batch, n_digits, n_ary)
            t_prob_c = torch.cat((t_prob_c_zero, (1.0-t_prob_c_zero)*t_prob_c), dim=-1)
        elif self._n_ary_internal == self._n_ary:
            # otherwise, simply adjust code probability so that it satisfies monotone increasing condition.
            t_prob_c = self.code_length_predictor(t_prob_c)
        else:
            raise ValueError(f"unexpected internal value: {self._n_ary_internal}")

        return t_prob_c

    @property
    def opts_internal_layers(self):
        ret = {}
        for attribute_names in ("_kwargs_stacked_lstm_layer", "_kwargs_code_length_predictor", "_kwargs_multi_dense_layer"):
            ret[attribute_names[1:]] = getattr(self, attribute_names, {})
        return ret

    @property
    def gate_open_ratio(self):
        return getattr(self.code_length_predictor, "gate_open_ratio", None)

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        if self.gate_open_ratio is not None:
            setattr(self.code_length_predictor, "gate_open_ratio", value)

    @property
    def use_built_in_discretizer(self):
        return False


class AutoRegressiveLSTMEncoder(SimpleEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 n_dim_hidden: Optional[int] = None,
                 n_dim_emb_code: Optional[int] = None,
                 discretizer: Optional[nn.Module] = None,
                 detach_previous_output: bool = False,
                 time_distributed: bool = True,
                 input_tranformation: str = "time_distributed",
                 output_embedding: str = "time_distributed",
                 prob_zero_monotone_increasing: bool = False,
                 dtype=torch.float32,
                 **kwargs):

        super(SimpleEncoder, self).__init__()

        self._discretizer = discretizer
        self._is_discretize_code_probability = discretizer is not None
        self._n_dim_emb = n_dim_emb
        self._n_dim_emb_code = int(n_digits*n_ary //2) if n_dim_emb_code is None else n_dim_emb_code
        self._n_dim_hidden = int(n_digits*n_ary //2) if n_dim_hidden is None else n_dim_hidden
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._dtype = dtype
        self._n_ary_internal = None
        self._time_distributed = time_distributed
        self._detach_previous_output = detach_previous_output
        self._input_transformation = input_tranformation
        self._output_embedding = output_embedding
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        self._build()

    def _build(self):

        # x -> i_t
        if self._input_transformation == "time_distributed": # i_t = FF(x;\theta)
            self._x_to_h = nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden)
        elif self._input_transformation == "time_dependent": # i_t = FF(x;\theta_t)
            lst_layers = [nn.Linear(in_features=self._n_dim_emb, out_features=self._n_dim_hidden, bias=True) for _ in range(self._n_digits)]
            self._x_to_h = nn.ModuleList(lst_layers)
        elif self._input_transformation == "none": # i_t = x
            assert self._n_dim_hidden == self._n_dim_emb, f"when you specify `input_transformation=none`, hidden dimension size must be consistent with embeddings dimension."
            self._x_to_h = nn.Identity()
        else:
            print(f"unknown `input_transformation` value: {self._input_transformation}")

        # (i_t,e_t,h_t) -> h_t
        self._lstm_cell = nn.LSTMCell(input_size=self._n_dim_hidden+self._n_dim_emb_code, hidden_size=self._n_dim_hidden, bias=True)

        # o_t -> e_{t+1}; t=0,1,...,N_d-2
        if self._output_embedding == "time_distributed": # e_t = Embed(o_t;\theta)
            self._embedding_code = nn.Linear(in_features=self._n_ary, out_features=self._n_dim_emb_code, bias=False)
        elif self._output_embedding == "time_dependent": # e_t = Embed(o_t;\theta_t)
            lst_layers = [nn.Linear(in_features=self._n_ary, out_features=self._n_dim_emb_code, bias=False) for _ in range(self._n_digits-1)]
            self._embedding_code = nn.ModuleList(lst_layers)
        else:
            print(f"unknown `input_transformation` value: {self._input_transformation}")

        # h_t -> z_t
        if self._time_distributed: # z_t = FF(h_t;\theta)
            self._lst_h_to_z = nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary, bias=True)
        else: # z_t = FF(h_t;\theta_t)
            lst_layers = [nn.Linear(in_features=self._n_dim_hidden, out_features=self._n_ary, bias=True) for _ in range(self._n_digits)]
            self._lst_h_to_z = nn.ModuleList(lst_layers)

    def init_bias_to_min(self):
        warnings.warn(f"it doesn't affect anything.")

    def init_bias_to_max(self):
        warnings.warn(f"it doesn't affect anything.")

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def _init_state(self, n_batch: int, dtype, device):
        h_t = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)
        c_t = torch.zeros((n_batch, self._n_dim_hidden), dtype=dtype, device=device)
        e_t = torch.zeros((n_batch, self._n_dim_emb_code), dtype=dtype, device=device)

        return h_t, c_t, e_t

    def forward(self, input_x: torch.Tensor, on_inference: bool = False):
        dtype, device = self._dtype_and_device(input_x)
        n_batch = input_x.shape[0]

        # initialize variables
        lst_prob_c = []
        lst_latent_code = []
        h_d, c_d, e_d = self._init_state(n_batch, dtype, device)

        # v_h: (n_batch, n_dim_hidden)
        if self._input_transformation == "time_distributed":
            t_h = torch.tanh(self._x_to_h(input_x))
        elif self._input_transformation == "none":
            t_h = self._x_to_h(input_x)

        for d in range(self._n_digits):
            if self._input_transformation in ("time_distributed", "none"):
                t_h_d = t_h
            elif self._input_transformation == "time_dependent":
                x_to_h_d = self._x_to_h[d]
                t_h_d = torch.tanh(x_to_h_d(input_x))

            input = torch.cat([t_h_d, e_d], dim=-1)
            h_d, c_d = self._lstm_cell(input, (h_d, c_d))

            # compute Pr{c_d=d|c_{<d}}
            if self._time_distributed:
                t_z_d_dash = self._lst_h_to_z(h_d)
            else:
                t_z_d_dash = self._lst_h_to_z[d](h_d)
            t_z_d = torch.log(F.softplus(t_z_d_dash) + 1E-6)
            t_prob_c_d = F.softmax(t_z_d, dim=-1)

            # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
            if self._prob_zero_monotone_increasing:
                prob_c_prev = lst_prob_c[-1] if len(lst_prob_c) > 0 else None
                t_prob_c_d = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

            # branch on training or on inference

            if on_inference:
                t_latent_code_d = t_prob_c_d
                # empirically, embedding based on the probability produces better result then argmax.
                # I guess it minimizes difference on training and inference.
                # t_latent_code_d = F.one_hot(t_prob_c_d.argmax(dim=-1), num_classes=self._n_ary).type(dtype)
            else:
                ## sample code
                if self._is_discretize_code_probability:
                    t_latent_code_d = self._discretizer(t_prob_c_d)
                else:
                    t_latent_code_d = t_prob_c_d

            # compute the embedding of previous code
            if d != (self._n_digits - 1):
                if self._detach_previous_output:
                    o_d = t_latent_code_d.detach()
                else:
                    o_d = t_latent_code_d
                if self._output_embedding == "time_distributed":
                    e_d = self._embedding_code(o_d)
                elif self._output_embedding == "time_dependent":
                    e_d = self._embedding_code[d](o_d)
                else:
                    e_d = None
            else:
                e_d = None

            # store computed results
            lst_prob_c.append(t_prob_c_d)
            lst_latent_code.append(t_latent_code_d)

        # stack code probability and latent code.
        # t_prob_c: (n_batch, n_digits, n_ary)
        t_prob_c = torch.stack(lst_prob_c, dim=1)
        # t_latent_code: (n_batch, n_digits, n_ary)
        t_latent_code = torch.stack(lst_latent_code, dim=1)

        return t_latent_code, t_prob_c

    def calc_code_probability(self, input_x: torch.Tensor, adjust_code_probability: bool = False, **kwargs):
        _, t_prob_c = self.forward(input_x, on_inference=True)
        if adjust_code_probability:
            t_prob_c = CodeValueMutualInformationLoss.calc_adjusted_code_probability(t_prob_c)

        return t_prob_c

    @property
    def use_built_in_discretizer(self):
        return self._is_discretize_code_probability

    @property
    def built_in_discretizer(self):
        return self._discretizer

    @property
    def gate_open_ratio(self):
        return None

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        pass



class TransformerEncoder(SimpleEncoder):

    def __init__(self, n_dim_emb: int, n_digits: int, n_ary: int,
                 normalize_digit_embeddings: bool,
                 n_layer: int = 4,
                 n_head: int = 4,
                 dropout: float = 0.1,
                 n_dim_emb_digits: Optional[int] = None,
                 time_distributed: bool = True,
                 how_digit_embeddings: str = "add",
                 prob_zero_monotone_increasing: bool = False,
                 dtype=torch.float32,
                 **kwargs):

        super(SimpleEncoder, self).__init__()

        self._n_dim_emb = n_dim_emb
        self._n_dim_emb_digits = n_dim_emb if n_dim_emb_digits is None else n_dim_emb_digits
        self._n_digits = n_digits
        self._n_ary = n_ary
        self._n_layer = n_layer
        self._n_head = n_head
        self._dropout = dropout
        self._dtype = dtype
        self._time_distributed = time_distributed
        self._how_digit_embeddings = how_digit_embeddings
        self._normalize_digit_embeddings = normalize_digit_embeddings
        self._prob_zero_monotone_increasing = prob_zero_monotone_increasing

        self._build()

    def _build(self):

        # digit embeddings
        cfg_embedding_layer = {
            "num_embeddings":self._n_digits,
            "embedding_dim":self._n_dim_emb_digits,
            "max_norm":1.0 if self._normalize_digit_embeddings else None
        }
        self._embedding_digit = nn.Embedding(**cfg_embedding_layer)

        if self._how_digit_embeddings == "add":
            n_dim_transformer = self._n_dim_emb
        elif self._how_digit_embeddings == "concat":
            n_dim_transformer = self._n_dim_emb + self._n_dim_emb_digits
        else:
            raise NotImplementedError(f"unknown `how_digit_embeddings` value: {self._how_digit_embeddings}")

        # transformer layers
        cfg_transformer_layer = {
            "d_model":n_dim_transformer,
            "nhead":self._n_head,
            "dim_feedforward":n_dim_transformer*4,
            "dropout":self._dropout
        }
        layer = nn.TransformerEncoderLayer(**cfg_transformer_layer)
        self._transformer = nn.TransformerEncoder(layer, num_layers=self._n_layer)

        # prediction layer
        if self._time_distributed:
            self._prediction_layer = nn.Linear(in_features=n_dim_transformer, out_features=self._n_ary, bias=True)
        else:
            lst_layers = [nn.Linear(in_features=n_dim_transformer, out_features=self._n_ary, bias=True) for _ in range(self._n_digits)]
            self._prediction_layer = nn.ModuleList(lst_layers)

    def init_bias_to_min(self):
        warnings.warn(f"it doesn't affect anything.")

    def init_bias_to_max(self):
        warnings.warn(f"it doesn't affect anything.")

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def forward(self, input_x: torch.Tensor, on_inference: bool = False):
        dtype, device = self._dtype_and_device(input_x)

        # input_x: (N_batch, N_dim_emb)
        n_batch = input_x.shape[0]

        # transformer layer inputs
        # x_in: (N_batch, N_digits, N_dim_transformer)

        ## entity embeddings
        x_in_entity = input_x.unsqueeze(1).repeat(1, self._n_digits, 1)
        ## digit embeddings
        dummy_digits = torch.arange(self._n_digits, dtype=torch.long, device=device)
        x_in_digits = self._embedding_digit.forward(dummy_digits).unsqueeze(0).repeat(n_batch,1,1)
        ## merge two embeddings
        if self._how_digit_embeddings == "add":
            x_in = x_in_entity + x_in_digits
        elif self._how_digit_embeddings == "concat":
            x_in = torch.cat((x_in_entity, x_in_digits), dim=-1)
        else:
            raise NotImplementedError(f"unknown `how_digit_embeddings` value: {self._how_digit_embeddings}")

        # transformer
        # transformer's input shape must be (N_digits, N_batch, N_dim_transformer)
        # h_out: (N_batch, N_digits, N_dim_transformer)
        h_out = self._transformer.forward(x_in.transpose(1,0)).transpose(0,1)

        # compute Pr{c_d}
        # t_prob_c: (n_batch, n_digits, n_ary)
        if self._time_distributed:
            lst_prob_c = [F.softmax(self._prediction_layer(h_out[:,idx_d,:]), dim=-1) for idx_d in range(self._n_digits)]
        else:
            lst_prob_c = [F.softmax(layer(h_out[:,idx_d,:]), dim=-1) for idx_d, layer in enumerate(self._prediction_layer)]

        # adjust Pr{c_d=d|c_{<d}} so that Pr{c_d=0|c_{<d+1}} satisfies monotone increasing condition.
        if self._prob_zero_monotone_increasing:
            for d in range(self._n_digits):
                prob_c_prev = lst_prob_c[d-1] if d > 0 else None
                t_prob_c_d = lst_prob_c[d]
                lst_prob_c[d] = self._adjust_code_probability_to_monotone_increasing(probs=t_prob_c_d, probs_prev=prob_c_prev)

        # stack each digits
        t_prob_c = torch.stack(lst_prob_c, dim=1)

        return t_prob_c

    def calc_code_probability(self, input_x: torch.Tensor, adjust_code_probability: bool = False, **kwargs):
        t_prob_c = self.forward(input_x, on_inference=True)
        if adjust_code_probability:
            t_prob_c = CodeValueMutualInformationLoss.calc_adjusted_code_probability(t_prob_c)

        return t_prob_c

    @property
    def gate_open_ratio(self):
        return None

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        pass
