"""
Reference:

- https://github.com/nju-websoft/muKG/blob/main/src/torch/kge_models/TransD.py
"""
import torch
import torch.nn as nn

from config import HyperParam
from base_model import KRLModel


class TransDHyperParam(HyperParam):
    ent_dim: int
    rel_dim: int
    norm: int
    margin: float


class TransD(KRLModel):
    def __init__(
        self,
        ent_num: int,
        rel_num: int,
        device: torch.device,
        hyper_params: TransDHyperParam
    ):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.norm = hyper_params.norm
        self.ent_dim = hyper_params.ent_dim
        self.rel_dim = hyper_params.rel_dim
        self.margin = hyper_params.margin
    
        self.margin_loss_fn = nn.MarginRankingLoss(margin=self.margin)
        
        # 初始化 ent_embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.ent_dim)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        
        # 初始化 rel_embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        
        # 初始化 transfer embedding
        self.ent_transfer = nn.Embedding(self.ent_num, self.ent_dim)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        self.rel_transfer = nn.Embedding(self.rel_num, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        
        self.dist_fn = nn.PairwiseDistance(p=self.norm)
    