"""
Calculate the metrics of KRL models.
"""
import torch


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
