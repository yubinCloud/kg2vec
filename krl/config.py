"""
用于填写配置信息的 Settings
"""

from pydantic import BaseModel, Field
from abc import ABC
from pathlib import Path
from typing import Optional, Mapping


######## Dataset config ########


class DatasetConf(BaseModel, ABC):
    dataset_name: str = Field(title='数据集的名称，方便打印时查看')


class BuiletinDatasetConf(DatasetConf):
    """
    内置的数据集的相关配置信息
    """
    source: str = Field(title='数据集的来源')


class BuiletinHuggingfaceDatasetConf(BuiletinDatasetConf):
    """
    存放于 Hugging Face datasets 上的数据集 
    :param BuiletinDatasetConf: _description_
    """
    huggingface_repo: str = Field(title='Hugging Face 中存放该数据集的 repo')


class LocalDatasetConf(DatasetConf):
    """
    存放于本地的数据集的相关配置信息
    """
    base_dir: Optional[Path] = Field(title='数据集的目录')
    entity2id_path: Optional[str] = Field(default='entity2id.txt', title='entity2id 的文件名')
    relation2id_path: Optional[str] = Field(default='relation2id.txt', title='relation2id 的文件名')
    train_path: Optional[str] = Field(default='train.txt', title='training set 的文件')
    valid_path: Optional[str] = Field(default='valid.txt', title='valid set 的文件')
    test_path: Optional[str] = Field(default='test.txt', title='testing set 的文件')


######## Dataset meta-data ########

class KRLDatasetMeta(BaseModel):
    entity2id: Mapping[str, int]
    rel2id: Mapping[str, int]


######## Hyper-parameters config ########


class HyperParam(BaseModel, ABC):
    """
    超参数，所有超参数的 Config 类都应该继承于它
    """
    batch_size: int = 128
    valid_batch_size: int = 64
    learning_rate: float = 0.001
    optimizer: str = Field(defualt='adam', title='optimizer name')
    epoch_size: int = 500
    valid_freq: int = Field(defualt=5, title='训练过程中，每隔多少次就做一次 valid 来验证是否保存模型')
    num_works: int = Field(default=64, title='`num_works` in dataloader')
    early_stoping_patience: int = Field(default=5, title='the patience of EarlyStoping')
    


######## Training config ########


class TrainConf(BaseModel):
    """
    训练的一些配置
    """
    logs_dir: Path = Field(title='The directory used to keep the log.')


######## Other HyperParam class for various models  ##########

class TransHyperParam(HyperParam):
    """
    Trans 系列模型的超参数类
    """
    embed_dim: int = 50
    norm: int = 1
    margin: int = 2.0
