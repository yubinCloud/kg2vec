from datasets import load_dataset
import json
from pathlib import Path

from .utils import convert_huggingface_dataset
from ..krl_dataset import BuiletinHuggingfaceDataset, KRLDatasetDict
from krl.config import BuiletinHuggingfaceDatasetConf



def __load_huggingface_krl_dataset(dataset_name: str) -> KRLDatasetDict:
    json_path = Path(__file__).parent / 'huggingface_krl_datasets_conf.json'
    with json_path.open() as f:
        name_to_confs = json.load(f)
    conf_dict = name_to_confs.get(dataset_name.lower())
    if conf_dict is None:
        raise NotImplemented(f"dataset {dataset_name} hasn't been implemented.")
    conf = BuiletinHuggingfaceDatasetConf(
        dataset_name=conf_dict['name'],
        source='HuggingFace',
        huggingface_repo=conf_dict['repo']
    )
    dataset_dict = load_dataset(conf.huggingface_repo)
    train_triples, valid_triples, test_triples, dataset_meta = convert_huggingface_dataset(dataset_dict)
    return KRLDatasetDict(
        train=BuiletinHuggingfaceDataset(conf, 'train', dataset_meta, train_triples),
        valid=BuiletinHuggingfaceDataset(conf, 'valid', dataset_meta, valid_triples),
        test=BuiletinHuggingfaceDataset(conf, 'test', dataset_meta, test_triples),
        meta=dataset_meta,
        dataset_conf=conf
    )


def load_krl_dataset(dataset_name: str):
    return __load_huggingface_krl_dataset(dataset_name)
