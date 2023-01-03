"""
Reference:

- https://yubincloud.github.io/notebook-paper/KG/KRL/1101.RESCAL-and-extensions.html
- https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/RESCAL.py
- https://github.com/nju-websoft/muKG/blob/main/src/torch/kge_models/RESCAL.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from base_model import KRLModel
from config import HyperParam


class RescalHyperParam(HyperParam):
    """Hyper-parameters of RESCAL
    """
    embed_dim: int
    alpha: float = Field(0.001, title='regularization parameter')


class RESCAL(KRLModel):
    def __init__(self,
                 ent_num: int,
                 rel_num: int,
                 device: torch.device,
                 embed_dim: int = 100,
                 alpha: float = 0.001):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.embed_dim = embed_dim
        self.alpha = alpha
        
        # initialize entity embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, 2, 1)  # 在许多实现中，这一行可以去掉
        
        # initialize relation embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim * self.embed_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, 2, 1)
    
        self.criterion = nn.MSELoss()
    
    def scoring(self, triples):
        """计算一个 batch 的三元组的 scores
        score 越大越好，正例接近 1，负例接近 0
        
        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
        assert triples.shape[1] == 3
        # get entity ids and relation ids
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        # id -> embedding
        h_embs = self.ent_embedding(heads)    # [batch, emb]
        t_embs = self.ent_embedding(tails)
        r_embs = self.rel_embedding(rels)     # [batch, emb * emb]
        # regualarization
        regul = (torch.mean(h_embs ** 2) + torch.mean(t_embs ** 2) + torch.mean(r_embs ** 2)) / 3
        # calcate scores
        r_embs = r_embs.view(-1, self.embed_dim, self.embed_dim)  # [batch, emb, emb]
        t_embs = t_embs.view(-1, self.embed_dim, 1)  # [batch, emb, 1]
        
        tr = torch.matmul(r_embs, t_embs)  # [batch, emb, 1]
        tr = tr.view(-1, self.embed_dim)  # [batch, emb]
        
        return torch.sum(h_embs * tr, dim=1), regul
    
    def loss(self, triples: torch.Tensor, labels: torch.Tensor):
        """Calculate the loss

        :param triples: a batch of triples. size: [batch, 3]
        :param labels: the label of each triple, label = 1 if the triple is positive, label = 0 if the triple is negative. size: [batch,]
        """
        assert triples.shape[1] == 3
        assert triples.shape[0] == labels.shape[0]
        
        scores, regul = self.scoring(triples)
        loss = self.criterion(scores, labels.float()) + self.alpha * regul
        
        return loss, scores
    
    def forward(self, triples, labels):
        loss, scores = self.loss(triples, labels)
        return loss, scores
    
    def predict(self, triples):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        batch_size = triples.size(0)
        labels = torch.ones([batch_size], device=self.device)
        _, scores = self.loss(triples, labels)
        return -scores
