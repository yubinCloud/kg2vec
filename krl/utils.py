import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple

from base_model import KRLModel
from dataset import DatasetConf, KRLDataset
from config import HyperParam

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


def create_dataloader(
    dataset_conf: DatasetConf,
    params: HyperParam,
    entity2id: Dict[str, int],
    rel2id: Dict[str, int]
) -> Tuple[KRLDataset, DataLoader, KRLDataset, DataLoader]:
    train_dataset = KRLDataset(dataset_conf, 'train', entity2id, rel2id)
    train_dataloader = DataLoader(train_dataset, params.batch_size)
    valid_dataset = KRLDataset(dataset_conf, 'valid', entity2id, rel2id)
    valid_dataloader = DataLoader(valid_dataset, params.valid_batch_size)
    return train_dataset, train_dataloader, valid_dataset, valid_dataloader