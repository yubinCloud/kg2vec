"""
The dataset class used to read KRL data, such as FB15k
"""

from typing import Literal, Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader

from config import DatasetConf


EntityMapping = Dict[str, int]
RelMapping = Dict[str, int]
Triple = List[int]

def create_mapping(dataset_conf: DatasetConf) -> Tuple[EntityMapping, RelMapping]:
    """
    create mapping of `entity2id` and `relation2id`
    """
    # 读取 entity2id
    entity2id = dict()
    entity2id_path = dataset_conf.base_dir / dataset_conf.entity2id_path
    if not entity2id_path.exists():
        raise FileNotFoundError(f'{entity2id_path} not found.')
    with entity2id_path.open() as f:
        for line in f:
            entity, entity_id = line.split()
            entity = entity.strip()
            entity_id = int(entity_id.strip())
            entity2id[entity] = entity_id
    # 读取 relation2id
    rel2id = dict()
    rel2id_path = dataset_conf.base_dir / dataset_conf.relation2id_path
    if not rel2id_path.exists():
        raise FileNotFoundError(f'{rel2id_path} not found.')
    with rel2id_path.open() as f:
        for line in f:
            rel, rel_id = line.split()
            rel = rel.strip()
            rel_id = int(rel_id.strip())
            rel2id[rel] = rel_id
    return entity2id, rel2id


class KRLDataset(Dataset):
    def __init__(self,
                 dataset_conf: DatasetConf,
                 mode: Literal['train', 'valid', 'test'],
                 entity2id: Dict[str, int],
                 rel2id: Dict[str, int]) -> None:
        super().__init__()
        self.conf = dataset_conf
        if mode not in {'train', 'valid', 'test'}:
            raise ValueError(f'dataset mode not support:{mode} mode')
        self.mode = mode
        self.triples = []
        self.entity2id = entity2id
        self.rel2id = rel2id
        self._read_triples()    # 读取数据集，并获得所有的 triples
    
    def _split_and_to_id(self, line: str) -> Triple:
        """将数据集文件中的一行数据进行切分，并将 entity 和 rel 转换成 id

        :param line: 数据集的一行数据
        :return: [head_id, rel_id, tail_id]
        """
        head, tail, rel = line.split()
        head_id = self.entity2id[head.strip()]
        rel_id = self.rel2id[rel.strip()]
        tail_id = self.entity2id[tail.strip()]
        return (head_id, rel_id, tail_id)
    
    def _read_triples(self):
        data_path = {
            'train': self.conf.train_path,
            'valid': self.conf.valid_path,
            'test': self.conf.test_path
        }.get(self.mode)
        p = self.conf.base_dir / data_path
        if not p.exists():
            raise FileNotFoundError(f'{p} not found.')
        with p.open() as f:
            self.triples = [self._split_and_to_id(line) for line in f]
    
    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.triples)
    
    def __getitem__(self, index) -> Triple:
        """Returns (head id, relation id, tail id)."""
        triple = self.triples[index]
        return triple[0], triple[1], triple[2]
    