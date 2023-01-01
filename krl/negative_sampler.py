"""
The sampler used to obtain negative samples for KRL.
"""
import torch
from abc import ABC, abstractmethod

from dataset import KRLDataset


class NegativeSampler(ABC):
    def __init__(self, dataset: KRLDataset, device: torch.device):
        self.dataset = dataset
        self.device = device
    
    @abstractmethod
    def neg_sample(self, heads, rels, tails):
        """执行负采样

        :param heads: 由 batch_size 个 head idx 组成的 tensor，size: [batch_size]
        :param rels: size [batch_size]
        :param tails: size [batch_size]
        """
        pass
    

class RandomNegativeSampler(NegativeSampler):
    """
    随机替换 head 或者 tail 来实现采样
    """
    def __init__(self, dataset: KRLDataset, device: torch.device):
        super().__init__(dataset, device)
        
    def neg_sample(self, heads, rels, tails):
        ent_num = len(self.dataset.entity2id)
        head_or_tail = torch.randint(high=2, size=heads.size(), device=self.device)
        random_entities = torch.randint(high=ent_num, size=heads.size(), device=self.device)
        corupted_heads = torch.where(head_or_tail == 1, random_entities, heads)
        corupted_tails = torch.where(head_or_tail == 0, random_entities, tails)
        return torch.stack([corupted_heads, rels, corupted_tails], dim=1)
