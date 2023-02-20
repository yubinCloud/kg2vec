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
    'sgd': optim.SGD,
    'adagrad': optim.Adagrad,
    'rms': optim.RMSprop
}

def create_optimizer(optimizer_name: str, model: KRLModel, lr: float) -> optim.Optimizer:
    """create a optimizer from a optimizer name
    """
    optim_klass = optimizer_map.get(optimizer_name)
    if optim_klass is None:
        raise NotImplementedError(f'No support for {optimizer_name} optimizer')
    return optim_klass(model.parameters(), lr)


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)


def create_local_dataloader(
    dataset_conf: LocalDatasetConf,
    params: HyperParam,
    entity2id: Mapping[str, int],
    rel2id: Mapping[str, int]
) -> Tuple[LocalKRLDataset, DataLoader, LocalKRLDataset, DataLoader]:
    train_dataset = LocalKRLDataset(dataset_conf, 'train', entity2id, rel2id)
    train_dataloader = DataLoader(train_dataset, params.batch_size)
    valid_dataset = LocalKRLDataset(dataset_conf, 'valid', entity2id, rel2id)
    valid_dataloader = DataLoader(valid_dataset, params.valid_batch_size)
    return train_dataset, train_dataloader, valid_dataset, valid_dataloader
