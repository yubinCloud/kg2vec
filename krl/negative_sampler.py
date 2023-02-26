"""
The sampler used to obtain negative samples for KRL.
"""
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional

from .dataset import KRLDataset


class NegativeSampler(ABC):
    def __init__(self, dataset: KRLDataset):
        self.dataset = dataset
    
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
    def __init__(self, dataset: KRLDataset):
        super().__init__(dataset)
        
    def neg_sample(self, heads, rels, tails):
        device = heads.device
        ent_num = len(self.dataset.entity2id)
        head_or_tail = torch.randint(high=2, size=heads.size(), device=device)
        random_entities = torch.randint(high=ent_num, size=heads.size(), device=device)
        corupted_heads = torch.where(head_or_tail == 1, random_entities, heads)
        corupted_tails = torch.where(head_or_tail == 0, random_entities, tails)
        return torch.stack([corupted_heads, rels, corupted_tails], dim=1)


class BernNegSampler(NegativeSampler):
    """
    Using bernoulli distribution to select whether to replace the head entity or tail entity.
    Specific sample process can refer to TransH paper or this implementation.
    """
    def __init__(self,
                 dataset: KRLDataset):
        """init function

        :param dataset: KRLDataset for negative sample
        :param device: device
        """
        super().__init__(dataset)
        self.entity2id = dataset.meta.entity2id
        self.rel2id = dataset.meta.rel2id
        self.ent_num = len(self.entity2id)
        self.rel_num = len(self.rel2id)

        self.probs_of_replace_head = self._cal_tph_and_hpt()  # 采样时替换 head 的概率
        assert self.probs_of_replace_head.shape[0] == self.rel_num
        
    def _cal_tph_and_hpt(self):
        dataloder = DataLoader(self.dataset, batch_size=1)
        r_h_matrix = torch.zeros([self.rel_num, self.ent_num])  # [i, j] 表示 r_i 与 h_j 有多少种尾实体
        r_t_matrxi = torch.zeros([self.rel_num, self.ent_num])  # [i, j] 表示 r_i 与 t_j 有多少种头实体
        for batch in iter(dataloder):
            h, r, t = batch[0], batch[1], batch[2]
            h = h.item()
            r = r.item()
            t = t.item()
            r_h_matrix[r, h] += 1
            r_t_matrxi[r, t] += 1
        tph = torch.sum(r_h_matrix, dim=1) / torch.sum(r_h_matrix != 0, dim=1)
        tph.nan_to_num_(1)  # 将 nan 填充为 1
        hpt = torch.sum(r_t_matrxi, dim=1) / torch.sum(r_t_matrxi != 0, dim=1)
        hpt.nan_to_num_(1)
        probs_of_replace_head = tph / (tph + hpt)
        probs_of_replace_head.nan_to_num_(0.5)
        return probs_of_replace_head
    
    def neg_sample(self, heads, rels, tails):
        device = heads.device
        batch_size = heads.shape[0]
        rands = torch.rand([batch_size])
        probs = self.probs_of_replace_head[rels.cpu()]
        head_or_tail = (rands < probs).to(device)  # True 的代表选择 head， False 的代表选择 tail
        random_entities = torch.randint(high=self.ent_num, size=heads.size(), device=device)
        corupted_heads = torch.where(head_or_tail == True, random_entities, heads)
        corupted_tails = torch.where(head_or_tail == False, random_entities, tails)
        return torch.stack([corupted_heads, rels, corupted_tails], dim=1)
        