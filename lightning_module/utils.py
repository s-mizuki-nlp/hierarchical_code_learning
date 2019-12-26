#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np

class BaseScheduler(object, metaclass=ABCMeta):

    def __init__(self, begin: float, end: float, step_boost: float = 1.0, **kwargs):

        self._begin = begin
        self._end = end
        self._step_boost = step_boost
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

    @abstractmethod
    def _eval(self, x: float) -> float:
        """
        returns scheduled value. when x=0, then y=start. similarly, when x=1, then y=stop.

        @param x: progress value. x \in [0,1].
        @return: y
        """

    def __call__(self, step):
        x = max(0.0, min(1.0, step*self._step_boost))
        return self._eval(x)


class LinearScheduler(BaseScheduler):

    def _eval(self, x: float) -> float:
        c = self._end - self._begin
        b = self._begin
        y = c*x + b
        return y


class ExponentialScheduler(BaseScheduler):

    def __init__(self, begin: float, end: float, step_boost: float = 1.0, **kwargs):

        super().__init__(begin, end, step_boost, **kwargs)

        assert (begin > 0) and (end > 0), f"both `begin` and `end` must be positive."


    def _eval(self, x: float) -> float:
        c = np.log(self._end) - np.log(self._begin)
        d = self._begin
        y = d*np.exp(c*x)
        return y


class SigmoidScheduler(BaseScheduler):

    def __init__(self, begin: float, end: float, step_boost: float = 1.0, coef_gamma: float = 10.0, **kwargs):

        super().__init__(begin, end, step_boost, **kwargs)
        self._coef_gamma = coef_gamma

    def _sigmoid(self, u: float):
        return 1./(1. + np.exp(-u))

    def _eval(self, x: float) -> float:
        s = self._sigmoid(self._coef_gamma*(x - 0.5))
        c = self._end - self._begin
        b = self._begin
        y = c*s + b
        return y


