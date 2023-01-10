from abc import ABC, abstractmethod
from typing import Any

from metric import KRLMetricBase, RankMetric
from config import DatasetConf


class MetricFormatter(ABC):
    """
    Every formatter class should derive this class.
    """
    @abstractmethod
    def convert(self, metric: KRLMetricBase) -> Any:
        """
        Convert metric model into a specific format

        :param metric: The metric instance that we want to convert.
        """
        pass


_STRING_TEMPLATE = """dataset: {dataset_name},
Hits@1: {hits_at_1},
Hits@3: {hits_at_3},
Hist@10: {hits_at_10},
MRR: {mrr}.
"""

class StringFormatter(MetricFormatter):
    """
    Convert the metric into string.
    """
    def convert(self, metric: RankMetric, dataset_conf: DatasetConf) -> str:
        return _STRING_TEMPLATE.format(
            dataset_name=dataset_conf.dataset_name,
            hits_at_1=metric.hits_at_1,
            hits_at_3=metric.hits_at_3,
            hits_at_10=metric.hits_at_10,
            mrr=metric.mrr
        )
