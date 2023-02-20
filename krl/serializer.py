from abc import ABC, abstractmethod

from .config import TrainConf, DatasetConf
from .metric import KRLMetricBase, RankMetric
from .metric_fomatter import MetricFormatter, StringFormatter


class BaseSerializer(ABC):
    """
    Serialize the metrics.
    Every serializer class should derive this class.
    """
    def __init__(self, train_conf: TrainConf, dataset_conf: DatasetConf) -> None:
        super().__init__()
        self.train_conf = train_conf
        self.dataset_conf = dataset_conf
    
    @abstractmethod
    def serialize(self, metric: KRLMetricBase, formatter: MetricFormatter) -> bool:
        pass


class FileSerializer(BaseSerializer):
    """
    Serilize the metric into local file.
    """
    def serialize(self, metric: RankMetric, formatter: StringFormatter) -> bool:
        """
        Serialize the string metric into file.

        :param metric: the metric instance that you want to serilize.
        :param formatter: We will use this formatter to convert metric instance into string.
        :return: success or not.
        """
        result = formatter.convert(metric, self.dataset_conf)
        with open(self.train_conf.metric_result_path, 'w') as f:
            f.write(result)
        return True
