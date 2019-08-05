#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class MutualInformationLoss(_Loss):

    def __init__(self):

        super(MutualInformationLoss, self).__init__()

