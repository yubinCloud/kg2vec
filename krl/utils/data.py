from torch.utils.data import DataLoader
from typing import Mapping, Tuple

from ..config import HyperParam, LocalDatasetConf
from ..dataset import LocalKRLDataset



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
