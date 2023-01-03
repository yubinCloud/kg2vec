"""
Reference:

- https://github.com/LYuhang/Trans-Implementation/blob/master/code/models/TransH.py
- https://github.com/zqhead/TransH/blob/master/TransH_torch.py 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field

from base_model import KRLModel
from config import TransHyperParam


class TransHHyperParam(TransHyperParam):
    """Hyper-paramters of TransH

    :param HyperParam: base hyper params model
    """
    C: float = Field(default=0.1, description='a hyper-parameter weighting the importance of soft constraints.')
    eps: float = Field(default=1e-3, description='the $\episilon$ in loss function')


class TransH(KRLModel):
    def __init__(self,
                 ent_num: int,
                 rel_num: int,
                 device: torch.device,
                 norm: int = 2,
                 embed_dim: int = 50,
                 margin: float = 1.0,
                 C: float = 1.0,
                 eps: float = 0.001):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.norm = norm
        self.embed_dim = embed_dim
        self.margin = margin
        self.C = C      # a hyper-parameter weighting the importance of soft constraints
        self.eps = eps  # the $\episilon$ in loss function
        
        self.margin_loss_fn = nn.MarginRankingLoss(margin=self.margin)
        
        # 初始化 ent_embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        
        # 初始化 rel_embedding，Embedding for the relation-specific translation vector $d_r$
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        
        # 初始化 rel_hyper_embedding，Embedding for the relation-specific hyperplane $w_r$
        self.rel_hyper_embedding = nn.Embedding(self.rel_num, self.embed_dim)
        nn.init.xavier_uniform_(self.rel_hyper_embedding.weight.data)
        
        self.dist_fn = nn.PairwiseDistance(p=self.norm) # the function for calculating the distance 
    
    def embed(self, triples):
        """get the embedding of triples

        :param triples: [heads, rels, tails]
        :return: embedding of triples.
        """
        assert triples.shape[1] == 3
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        h_embs = self.ent_embedding(heads)  # h_embs: [batch, embed_dim]
        t_embs = self.ent_embedding(tails)
        r_embs = self.rel_embedding(rels)
        r_hyper_embs = self.rel_hyper_embedding(rels)  # relation hyperplane, size: [batch_size, embed_dim]
        return h_embs, r_embs, t_embs, r_hyper_embs
    
    def _project(self, ent_embeds, rel_hyper_embeds):
        """Project entity embedding into relation hyperplane
        computational process: $h - w_r^T h w_r$

        :param ent_embeds: entity embedding, size: [batch_size, embed_dim]
        :param rel_hyper_embeds: relation hyperplane, size: [batch_size, embed_dim]
        """
        return ent_embeds - rel_hyper_embeds * torch.sum(ent_embeds * rel_hyper_embeds, dim=1, keepdim=True)
    
    def _distance(self, triples):
        """计算一个 batch 的三元组的 distance

        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
        assert triples.shape[1] == 3
        # step 1: Transform index tensor to embedding tensor.
        h_embs, r_embs, t_embs, r_hyper_embs = self.embed(triples)
        # step 2: Project entity head and tail embedding to relation hyperplane
        h_embs = self.project(h_embs, r_hyper_embs)
        t_embs = self.project(t_embs, r_hyper_embs)
        # step 3: Calculate similarity score in relation hyperplane
        return self.dist_fn(h_embs + r_embs, t_embs)
    
    def _cal_margin_based_loss(self, pos_distances, neg_distances):
        """Calculate the margin-based loss

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: margin_based loss, size: [1,]
        """
        ones = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.margin_loss_fn(pos_distances, neg_distances, ones)

    def _cal_scale_loss(self):
        """Calculate the scale loss.
        F.relu(x) is equal to max(x, 0).
        """
        ent_norm = torch.norm(self.ent_embedding.weight, p=2, dim=1)  # the L2 norm of entity embedding, size: [ent_num, ]
        scale_loss = torch.sum(F.relu(ent_norm - 1))
        return scale_loss
    
    def _cal_orthogonal_loss(self):
        """Calculate the orthogonal loss.
        """
        orth_loss = torch.sum(F.relu(torch.sum(self.rel_hyper_embedding.weight * self.rel_embedding.weight, dim=1, keepdim=False) / torch.norm(self.rel_embedding.weight, p=2, dim=1, keepdim=False) - self.eps ** 2))
        return orth_loss
    
    def loss(self, pos_distances, neg_distances):
        """Calculate the loss

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: loss
        """
        margin_based_loss = self._cal_margin_based_loss(pos_distances, neg_distances)
        scale_loss = self._cal_scale_loss()
        orth_loss = self._cal_orthogonal_loss()
        ent_num = self.ent_num
        return margin_based_loss + self.C * (scale_loss / ent_num + orth_loss / ent_num)
    
    def forward(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor):
        """Return model losses based on the input.

        :param pos_triples: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param neg_triples: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        assert pos_triples.size()[1] == 3
        assert neg_triples.size()[1] == 3
        
        pos_distances = self._distance(pos_triples)
        neg_distances = self._distance(neg_triples)
        loss = self.loss(pos_distances, neg_distances)
        return loss, pos_distances, neg_distances

    def predict(self, triples: torch.Tensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triples)
