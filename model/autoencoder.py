#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import List, Optional
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from contextlib import ExitStack

class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discretizer: nn.Module, normalize_output_length: bool = False, dtype=torch.float32, **kwargs):

        super(AutoEncoder, self).__init__()
        self._encoder = encoder
        self._encoder_class_name = encoder.__class__.__name__
        self._decoder = decoder

        built_in_discretizer = getattr(self._encoder, "use_built_in_discretizer", False)
        if built_in_discretizer:
            self._discretizer = self._encoder.built_in_discretizer
        else:
            self._discretizer = discretizer
        self._normalize_output_length = normalize_output_length
        self._dtype = dtype

    @property
    def n_ary(self):
        return self._encoder._n_ary

    @property
    def n_digits(self):
        return self._encoder._n_digits

    @property
    def temperature(self):
        return getattr(self._discretizer, "temperature", None)

    @temperature.setter
    def temperature(self, value):
        if self.temperature is not None:
            setattr(self._discretizer, "temperature", value)

    @property
    def gate_open_ratio(self):
        value_e = getattr(self._encoder, "gate_open_ratio", None)
        value_d = getattr(self._discretizer, "gate_open_ratio", None)
        if value_e is not None:
            return value_e
        elif value_d is not None:
            return value_d
        else:
            return None

    @gate_open_ratio.setter
    def gate_open_ratio(self, value):
        if hasattr(self._encoder, "gate_open_ratio"):
            setattr(self._encoder, "gate_open_ratio", value)
        if hasattr(self._discretizer, "gate_open_ratio"):
            setattr(self._discretizer, "gate_open_ratio", value)

    def _numpy_to_tensor(self, np_array: np.array):
        return torch.from_numpy(np_array).type(self._dtype)

    def _tensor_to_numpy(self, t_x: torch.Tensor):
        return t_x.cpu().numpy()

    def _normalize(self, x: torch.Tensor, x_dash: torch.Tensor):
        """
        adjust reconstructed embeddings (`x_dash`) so that the L2 norm will be identical to the original embeddings (`x`).

        :param x: original embeddings
        :param x_dash: reconstructed embeddings
        :return: length-normalized reconstructed embeddings
        """
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_dash_norm = torch.norm(x_dash, dim=-1, keepdim=True)
        scale_factor = x_norm / (x_dash_norm + 1E-7)

        return x_dash * scale_factor

    def forward(self, t_x: torch.Tensor, requires_grad: bool = True, enable_discretizer: bool = True):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            # encoder and discretizer
            if self._encoder_class_name == "AutoRegressiveLSTMEncoder":
                if self._encoder.use_built_in_discretizer:
                    t_latent_code, t_code_prob = self._encoder.forward(t_x)
                else:
                    _, t_code_prob = self._encoder.forward(t_x)
                    if enable_discretizer:
                        t_latent_code = self._discretizer.forward(t_code_prob)
                    else:
                        t_latent_code = t_code_prob
            else:
                t_code_prob = self._encoder.forward(t_x)
                if enable_discretizer:
                    t_latent_code = self._discretizer.forward(t_code_prob)
                else:
                    t_latent_code = t_code_prob

            # decoder
            t_x_dash = self._decoder.forward(t_latent_code)

            # length-normalizer
            if self._normalize_output_length:
                t_x_dash = self._normalize(x=t_x, x_dash=t_x_dash)

        return t_latent_code, t_code_prob, t_x_dash

    def _predict(self, t_x: torch.Tensor):

        return self.forward(t_x, requires_grad=False, enable_discretizer=False)

    def predict(self, mat_x: np.ndarray):

        t_x = self._numpy_to_tensor(mat_x)
        t_latent_code, t_code_prob, t_x_dash = self._predict(t_x)

        return tuple(map(self._tensor_to_numpy, (t_latent_code, t_code_prob, t_x_dash)))

    def _encode(self, t_x: torch.Tensor, **kwargs):

        t_code_prob = self._encoder.calc_code_probability(t_x, **kwargs)
        t_code = torch.argmax(t_code_prob, dim=2, keepdim=False)
        return t_code

    def encode(self, mat_x: np.ndarray, **kwargs):

        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            t_x = self._numpy_to_tensor(mat_x)
            t_code = self._encode(t_x, **kwargs)

        return t_code.cpu().numpy()

    def _encode_soft(self, t_x: torch.Tensor, **kwargs):
        t_code_prob = self._encoder.calc_code_probability(t_x, **kwargs)
        return t_code_prob

    def encode_soft(self, mat_x: np.ndarray, **kwargs):

        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            t_x = self._numpy_to_tensor(mat_x)
            t_prob = self._encode_soft(t_x, **kwargs)

        return t_prob.cpu().numpy()

    def _decode(self, t_code_prob: torch.Tensor):

        # t_code_prob: (N_batch, N_digits, N_ary), t_c_[b,n,m] \in [0,1]
        return self._decoder.forward(t_code_prob)

    def decode(self, mat_code: np.ndarray):

        with ExitStack() as context_stack:
            context_stack.enter_context(torch.no_grad())
            n_ary = self._decoder.n_ary
            # one-hot encoding
            t_code_prob = self._numpy_to_tensor(np.eye(n_ary)[mat_code])
            t_x_dash = self._decode(t_code_prob)

        return t_x_dash.cpu().numpy()


class MaskedAutoEncoder(AutoEncoder):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discretizer: nn.Module, masked_value: int = 0, normalize_output_length: bool = True, dtype=torch.float32, **kwargs):
        """

        :param encoder:
        :param decoder:
        :param discretizer:
        :param masked_value: the code value that is masked (=ignored) during decoding process.
            currently valid value is `0`, otherwise it will raise error.
            in future, it may accept multiple values or digit-dependent values.
        :param normalize_output_length:
        :param dtype:
        :param kwargs:
        """
        if not normalize_output_length:
            warnings.warn("it is recommended to enable output length normalization.")

        super(MaskedAutoEncoder, self).__init__(encoder, decoder, discretizer, normalize_output_length, dtype)

        assert masked_value == 0, "currently `mased_value` must be `0`, otherwise it will raise error."
        self._masked_value = masked_value

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def _build_mask_tensor(self, masked_value: int, dtype, device):

        mask_shape = (1, self.n_digits, self.n_ary)
        mask_tensor = torch.ones(mask_shape, dtype=dtype, device=device, requires_grad=False)
        mask_tensor[:,:,masked_value] = 0.0

        return mask_tensor

    def forward(self, t_x: torch.Tensor, requires_grad: bool = True, enable_discretizer: bool = True):

        with ExitStack() as context_stack:
            # if user doesn't require gradient, disable back-propagation
            if not requires_grad:
                context_stack.enter_context(torch.no_grad())

            # encoder and discretizer
            if self._encoder_class_name == "AutoRegressiveLSTMEncoder":
                if self._encoder.use_built_in_discretizer:
                    t_latent_code, t_code_prob = self._encoder.forward(t_x)
                else:
                    _, t_code_prob = self._encoder.forward(t_x)
                    if enable_discretizer:
                        t_latent_code = self._discretizer.forward(t_code_prob)
                    else:
                        t_latent_code = t_code_prob
            else:
                t_code_prob = self._encoder.forward(t_x)
                if enable_discretizer:
                    t_latent_code = self._discretizer.forward(t_code_prob)
                else:
                    t_latent_code = t_code_prob

            # mask intermediate representation
            dtype, device = self._dtype_and_device(t_x)
            mask = self._build_mask_tensor(self._masked_value, dtype, device)
            t_decoder_input = t_latent_code * mask

            # decoder
            t_x_dash = self._decoder.forward(t_decoder_input)

            # length-normalizer
            if self._normalize_output_length:
                t_x_dash = self._normalize(x=t_x, x_dash=t_x_dash)

        return t_latent_code, t_code_prob, t_x_dash
