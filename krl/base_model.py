"""
Base class for various KRL models.
"""

import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from typing import Any


class KRLModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def embed(self, triples):
        """get the embeddings of the triples

        :param triples: a batch of triples, which consists of (heads, rels, tails) id.
        :return: the embedding of (heads, rels, tails)
        """
        pass
    
    @abstractmethod
    def forward(self):
        """Return model losses based on the input.
        """
        pass
    
    @abstractmethod
    def predict(self, triples: torch.Tensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        pass


class ModelMain(ABC):
    """Act as a main function for a KRL model.
    """
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __call__(self):
        pass
