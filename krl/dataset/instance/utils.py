from datasets.dataset_dict import DatasetDict as HuggingfaceDatasetDict
from datasets.arrow_dataset import Dataset as HuggingFaceDataset
from typing import Tuple, List

from ..krl_dataset import BuiletinHuggingfaceDataset
from ...config import KRLDatasetMeta


Triple = Tuple[int, int, int]
Triples = List[Triple]

def convert_huggingface_dataset(
    dataset_dict: HuggingfaceDatasetDict
) -> Tuple[Triples, Triples, Triples, KRLDatasetMeta]:
    entity2id = {}
    rel2id = {}
    
    def read_dataset(dataset: HuggingFaceDataset):
        triples = []
        for triple in dataset:
            head, rel, tail = str(triple['head']), str(triple['relation']), str(triple['tail'])
            if head not in entity2id:
                entity2id[head] = len(entity2id)
            head_id = entity2id[head]
            if tail not in entity2id:
                entity2id[tail] = len(entity2id)
            tail_id = entity2id[tail]
            if rel not in rel2id:
                rel2id[rel] = len(rel2id)
            rel_id = rel2id[rel]
            triples.append((head_id, rel_id, tail_id))
        return triples
    
    train_triples = read_dataset(dataset_dict['train'])
    valid_triples = read_dataset(dataset_dict['validation'])
    test_triples = read_dataset(dataset_dict['test'])
    
    dataset_meta = KRLDatasetMeta(
        entity2id=entity2id,
        rel2id=rel2id
    )
    
    return train_triples, valid_triples, test_triples, dataset_meta

