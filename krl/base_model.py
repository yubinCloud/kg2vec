"""
Base class for various KRL models.
"""

import torch.nn as nn
import torch
from abc import ABC, abstractclassmethod


class KRLModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractclassmethod
    def loss(self, pos_distances, neg_distances) -> torch.Tensor:
        """计算模型的损失

        :param pos_distances: _description_
        :param neg_distances: _description_
        """
        pass
    
    @abstractclassmethod
    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor):
        """Return model losses based on the input.

        :param pos_triples: _description_
        :param neg_triples: _description_
        """
        pass
    
    @abstractclassmethod
    def predict(self, triples: torch.Tensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        pass
