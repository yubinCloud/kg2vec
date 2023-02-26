import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Mapping
import numpy as np
import random

from .base_model import KRLModel
from .dataset import LocalKRLDataset
from .config import HyperParam, LocalDatasetConf

optimizer_map = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'radam': optim.RAdam,
    'sgd': optim.SGD,
    'adagrad': optim.Adagrad,
    'rms': optim.RMSprop,
}

def create_optimizer(optimizer_name: str, model: KRLModel, lr: float) -> optim.Optimizer:
    """create a optimizer from a optimizer name
    """
    optim_klass = optimizer_map.get(optimizer_name.lower())
    if optim_klass is None:
        raise NotImplementedError(f'No support for {optimizer_name} optimizer')
    return optim_klass(model.parameters(), lr)
