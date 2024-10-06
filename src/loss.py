# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss.py
# Time       ：27/3/2024 2:49 pm
# Author     ：XXXXXX
# version    ：python 
# Description：一些特殊的loss
"""
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class RqVaeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query, value):
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1, -2])
        query_loss = ((query - value.detach())**2).sum(axis=[-1, -2])
        return emb_loss + self.commitment_weight * query_loss


