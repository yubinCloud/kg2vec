import torch
from abc import ABC, abstractmethod
from typing import List

from metric import KRLMetricBase, KRLMetric, MetricEnum


def cal_hits_at_k(predictions: torch.Tensor,
                  ground_truth_idx: torch.Tensor,
                  device: torch.device,
                  k: int) -> float:
    """Calculates number of hits@k.

    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :param k: number of top K results to be considered as hits
    :return: Hits@K scoreH
    """
    assert predictions.size()[0] == ground_truth_idx.size()[0]  # has the same batch_size
    
    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = predictions.topk(k, largest=False)  # indices: [batch_size, k]
    where_flags = indices == ground_truth_idx  # where_flags: [batch_size, k], type: bool
    hits = torch.where(where_flags, one_tensor, zero_tensor).sum().item()
    return hits

def cal_mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.

    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()



class Evaluator(ABC):
    """
    Every evaluator should derive this base class.
    """
    
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
    
    @abstractmethod
    def evaluate(
        self,
        predictions: torch.Tensor,
        ground_truth_idx: torch.Tensor
    ) -> KRLMetricBase:
        """
        Calculate the metrics of model's prediction.

        :param predictions: _description_
        :param ground_truth_idx: _description_
        :return: _description_
        """
        pass


class KRLEvaluator(Evaluator):
    
    _SUPPORT_METRICS = {
        MetricEnum.MRR,
        MetricEnum.HITS_AT_1,
        MetricEnum.HITS_AT_3,
        MetricEnum.HITS_AT_10
    }
    
    def __init__(
        self,
        device: torch.device,
        metrics: List[MetricEnum]
    ) -> None:
        """
        :param metrics: The metrics that you want to calcualte.
        """
        super().__init__(device)
        for m in metrics:
            if m not in KRLEvaluator._SUPPORT_METRICS:
                raise NotImplementedError(f"Evaluator don't support metric: {m.value}")
        self.metrics = set(metrics)
    
    def evaluate(
        self,
        predictions: torch.Tensor,
        ground_truth_idx: torch.Tensor
    ) -> KRLMetric:
        metric = KRLMetric()
        
        if MetricEnum.MRR in self.metrics:
            metric.mrr = cal_mrr(predictions, ground_truth_idx)
        if MetricEnum.HITS_AT_1 in self.metrics:
            metric.hits_at_1 = cal_hits_at_k(predictions, ground_truth_idx, self.device, 1)
        if MetricEnum.HITS_AT_3 in self.metrics:
            metric.hits_at_3 = cal_hits_at_k(predictions, ground_truth_idx, self.device, 3)
        if MetricEnum.HITS_AT_10 in self.metrics:
            metric.hits_at_10 = cal_hits_at_k(predictions, ground_truth_idx, self.device, 10)

        return metric