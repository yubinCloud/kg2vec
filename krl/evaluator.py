import torch
from abc import ABC, abstractmethod
from typing import List

from metric import KRLMetricBase, RankMetric, MetricEnum


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
    
    @abstractmethod
    def clear(self):
        """
        Clear this evaluator for reusing.
        """
        pass
    
    @abstractmethod
    def reset_metrics(self, metrics: List[MetricEnum]):
        """
        reset the metrics that you want to calculate.
        
        :param metrics: the metric list.
        :type metrics: List[MetricEnum]
        """
        pass
    
    def export_metrics(self) -> KRLMetricBase:
        """
        export the metric result stored in evaluator.
        """
        pass


class RankEvaluator(Evaluator):
    
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
        self.example_cnt = 0
        self.metrics = None
        self._mrr_sum = None
        self._hits_at_1_sum = None
        self._hits_at_3_sum = None
        self._hits_at_10_sum = None
        # checks the metrics that you want to calcualte
        self.reset_metrics(metrics)
        # set to 0 if you want to calcualte this metric
        self.clear()
    
    def clear(self):
        self.example_cnt = 0
        self._mrr_sum = None if MetricEnum.MRR not in self.metrics else 0
        self._hits_at_1_sum = None if MetricEnum.HITS_AT_1 not in self.metrics else 0
        self._hits_at_3_sum = None if MetricEnum.HITS_AT_3 not in self.metrics else 0
        self._hits_at_10_sum = None if MetricEnum.HITS_AT_10 not in self.metrics else 0
    
    def reset_metrics(self, metrics: List[MetricEnum]):
        for m in metrics:
            if m not in RankEvaluator._SUPPORT_METRICS:
                raise NotImplementedError(f"Evaluator don't support metric: {m.value}")
        self.metrics = set(metrics)

    def evaluate(
        self,
        predictions: torch.Tensor,
        ground_truth_idx: torch.Tensor
    ):
        self.example_cnt += predictions.size(0)
        if MetricEnum.MRR in self.metrics:
            self._mrr_sum += cal_mrr(predictions, ground_truth_idx)
        if MetricEnum.HITS_AT_1 in self.metrics:
            self._hits_at_1_sum += cal_hits_at_k(predictions, ground_truth_idx, self.device, 1)
        if MetricEnum.HITS_AT_3 in self.metrics:
            self._hits_at_3_sum += cal_hits_at_k(predictions, ground_truth_idx, self.device, 3)
        if MetricEnum.HITS_AT_10 in self.metrics:
            self._hits_at_10_sum += cal_hits_at_k(predictions, ground_truth_idx, self.device, 10)
    
    def export_metrics(self) -> RankMetric:
        """
        Export the metric result stored in evaluator
        """
        result = RankMetric(
            mrr=None if MetricEnum.MRR not in self.metrics else self._percentage(self._mrr_sum),
            hits_at_1=None if MetricEnum.HITS_AT_1 not in self.metrics else self._percentage(self._hits_at_1_sum),
            hits_at_3=None if MetricEnum.HITS_AT_3 not in self.metrics else self._percentage(self._hits_at_3_sum),
            hits_at_10=None if MetricEnum.HITS_AT_10 not in self.metrics else self._percentage(self._hits_at_10_sum)
        )
        return result

    def _percentage(self, sum):
        return sum / self.example_cnt * 100