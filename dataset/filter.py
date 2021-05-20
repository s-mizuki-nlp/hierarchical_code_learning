#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


from typing import Optional, Dict, Set, Callable, Any


class DictionaryFilter(object):

    def __init__(self, includes: Optional[Dict[str, Set]] = None,
                 excludes: Optional[Dict[str, Set]] = None):
        self._includes = includes
        self._excludes = excludes
        assert (includes is not None) or (excludes is not None), f"you must specify either `includes` or `excludes` argument."
        assert (includes is None) or (excludes is None), f"you can't specify both `includes` and `excludes` at the same time."

    def __call__(self, sample: Dict[str, str]):

        if self._includes is not None:
            for field_name, values in self._includes.items():
                if sample[field_name] in values:
                    return True
            return False

        if self._excludes is not None:
            for field_name, values in self._excludes.items():
                if sample[field_name] in values:
                    return False
            return True


class FunctionFilter(object):

    def __init__(self, include: Optional[Callable[[Dict[str, str]], bool]] = None,
                 exclude: Optional[Callable[[Dict[str, str]], bool]] = None):
        self._include = include
        self._exclude = exclude
        assert (include is not None) or (exclude is not None), f"you must specify either `includes` or `excludes` argument."
        assert (include is None) or (exclude is None), f"you can't specify both `includes` and `excludes` at the same time."

    def __call__(self, sample: Dict[str, str]):

        if self._include is not None:
            if self._include(sample) == True:
                return True
            else:
                return False

        if self._exclude is not None:
            if self._exclude(sample) == True:
                return False
            else:
                return True