"""
Calculate the metrics of KRL models.
"""
from pydantic import BaseModel
from typing import Optional
from abc import ABC
from enum import Enum
import torchmetrics
import torch


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


class RankMetric(KRLMetricBase):
    mrr: Optional[float]
    hits_at_1: Optional[float]
    hits_at_3: Optional[float]
    hits_at_10: Optional[float]


class MRR(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("mrr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("example_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _cal_mrr(self, predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
        """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.

        :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
        must be sorted in class ids order
        :param ground_truth_idx: Bx1 tensor with index of ground truth class
        :return: Mean reciprocal rank score
        """
        assert predictions.size(0) == ground_truth_idx.size(0)

        indices = predictions.argsort()
        return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.mrr_sum += self._cal_mrr(preds, target)
        self.example_cnt += preds.size(0)
    
    def compute(self):
        return self.mrr_sum.float() / self.example_cnt * 100
    

class HitsAtK(torchmetrics.Metric):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("hits_sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("example_cnt", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def _cal_hits_at_k(
        self,        
        predictions: torch.Tensor,
        ground_truth_idx: torch.Tensor
    ) -> float:
        """Calculates number of hits@k.

        :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
        must be sorted in class ids order
        :param ground_truth_idx: Bx1 tensor with index of ground truth class
        :param k: number of top K results to be considered as hits
        :return: Hits@K scoreH
        """
        assert predictions.size()[0] == ground_truth_idx.size()[0]  # has the same batch_size

        device = predictions.device
        
        zero_tensor = torch.tensor([0], device=device)
        one_tensor = torch.tensor([1], device=device)
        _, indices = predictions.topk(self.k, largest=False)  # indices: [batch_size, k]
        where_flags = indices == ground_truth_idx  # where_flags: [batch_size, k], type: bool
        hits = torch.where(where_flags, one_tensor, zero_tensor).sum().item()
        return hits

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.hits_sum += self._cal_hits_at_k(preds, target)
        self.example_cnt += preds.size(0)
    
    def compute(self) -> float:
        return self.hits_sum.float() / self.example_cnt * 100
