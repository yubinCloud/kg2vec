"""
用于填写配置信息的 Settings
"""

from pydantic import BaseSettings, BaseModel, Field
from abc import ABC


class DatasetConf(BaseSettings):
    """
    数据集的相关配置信息
    """
    dataset_name: str = Field(title='数据集的名称，方便打印时查看')
    base_dir: str = Field(title='数据集的目录')
    entity2id_path: str = Field(default='/entity2id.txt', title='entity2id 的文件名')
    relation2id_path: str = Field(default='/relation2id.txt', title='relation2id 的文件名')
    train_path: str = Field(default='/train.txt', title='training set 的文件')
    valid_path: str = Field(default='/valid.txt', title='valid set 的文件')
    test_path: str = Field(default='/test.txt', title='testing set 的目录')


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


class TrainConf(BaseModel):
    """
    训练的一些配置
    """
    checkpoint_path: str = Field(title='保存模型的路径')
    metric_result_path: str = Field(title='运行 test 的 metric 输出位置')


######## Other HyperParam class for various models  ##########

class TransHyperParam(HyperParam):
    """
    Trans 系列模型的超参数类
    """
    embed_dim: int = 50
    norm: int = 1
    margin: int = 2.0
