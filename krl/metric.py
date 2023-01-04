"""
Calculate the metrics of KRL models.
"""
from pydantic import BaseModel
from typing import Optional
from abc import ABC
from enum import Enum


class MetricEnum(Enum):
    """
    Enumerate all metric name. This name is the attribute of Metric Model which derived from `KRLMetricBase`.
    """
    MRR = 'mrr'
    HITS_AT_1 = 'hits_at_1'
    HITS_AT_3 = 'hits_at_3'
    HITS_AT_10 = 'hits_at_10'


class KRLMetricBase(BaseModel, ABC):
    """All metric model class should derive this class.
    """
    pass


class KRLMetric(KRLMetricBase):
    mrr: Optional[float]
    hits_at_1: Optional[float]
    hits_at_3: Optional[float]
    hits_at_10: Optional[float]
