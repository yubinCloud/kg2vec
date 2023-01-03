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
    def loss(self) -> torch.Tensor:
        """计算模型的损失
        """
        pass
    
    @abstractclassmethod
    def forward(self):
        """Return model losses based on the input.
        """
        pass
    
    @abstractclassmethod
    def predict(self, triples: torch.Tensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        pass
