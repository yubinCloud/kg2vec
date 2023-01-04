"""
Reference:

- https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransR.py
- https://github.com/zqhead/TransR
- https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/models/pairwise.py
- https://github.com/nju-websoft/muKG/blob/main/src/torch/kge_models/TransR.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import KRLModel
from config import HyperParam


class TransRHyperParam(HyperParam):
    """Hyper-parameters of TransR
    """
    ent_dim: int
    rel_dim: int
    norm: int
    margin: float
    C: float
    

class TransR(KRLModel):
    def __init__(
        self,
        ent_num: int,
        rel_num: int,
        device: torch.device,
        hyper_params: TransRHyperParam
    ):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        
        self.ent_dim = hyper_params.ent_dim
        self.rel_dim = hyper_params.rel_dim
        self.norm = hyper_params.norm
        self.margin = hyper_params.margin
        self.C = hyper_params.C
        
        # initialize ent_embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.ent_dim)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        
        # initialize rel_embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        
        # initialize trasfer matrix
        self.transfer_matrix = nn.Embedding(self.rel_num, self.ent_dim * self.rel_dim)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
        
        self.margin_loss_fn = nn.MarginRankingLoss(margin=self.margin)
        self.dist_fn = nn.PairwiseDistance(p=self.norm)
        
    def _transfer(self, ent_embs: torch.Tensor, rels_tranfer: torch.Tensor):
        """Tranfer the entity space into the relation-specfic space

        :param ent_embs: [batch, ent_dim]
        :param rels_tranfer: [batch, ent_dim * rel_dim]
        """
        assert ent_embs.size(0) == rels_tranfer.size(0)
        assert ent_embs.size(1) == self.ent_dim
        assert rels_tranfer.size(1) == self.ent_dim * self.rel_dim
            
        rels_tranfer = rels_tranfer.reshape(-1, self.ent_dim, self.rel_dim)    # [batch, ent_dim, rel_dim]
        ent_embs = ent_embs.reshape(-1, 1, self.ent_dim)   # [batch, 1, ent_dim]
            
        ent_proj = torch.matmul(ent_embs, rels_tranfer)   # [batch, 1, rel_dim]
        return ent_proj.reshape(-1, self.rel_dim)   # [batch, rel_dim]
        
    def embed(self, triples):
        """get the embedding of triples
            
        :param triples: [heads, rels, tails]
        :return: embedding of triples.
        """
        assert triples.size(1) == 3
        # split triples
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        # id -> embedding
        h_embs = self.ent_embedding(heads)  # h_embs: [batch, embed_dim]
        t_embs = self.ent_embedding(tails)
        r_embs = self.rel_embedding(rels)
        rels_tranfer = self.transfer_matrix(rels)
        # tranfer the space
        h_embs = self._transfer(h_embs, rels_tranfer)
        t_embs = self._transfer(t_embs, rels_tranfer)
        return h_embs, r_embs, t_embs
    
    def _distance(self, triples):
        """计算一个 batch 的三元组的 distance

        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
        assert triples.shape[1] == 3
        # id -> embedding
        h_embs, r_embs, t_embs = self.embed(triples)
        return self.dist_fn(h_embs + r_embs, t_embs)
    
    def _cal_margin_base_loss(self, pos_distances, neg_distances):
        """Calculate the margin-based loss

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: margin_based loss, size: [1,]
        """
        ones = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.margin_loss_fn(pos_distances, neg_distances, ones)
    
    def _cal_scale_loss(self, embedding: nn.Embedding):
        """Calculate the scale loss.
        F.relu(x) is equal to max(x, 0).
        """
        norm = torch.norm(embedding.weight, p=2, dim=1)  # the L2 norm of entity embedding, size: [ent_num, ]
        scale_loss = torch.sum(F.relu(norm - 1))
        return scale_loss
    
    def loss(self, pos_distances, neg_distances):
        """Calculate the loss

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: loss
        """
        margin_based_loss = self._cal_margin_base_loss(pos_distances, neg_distances)
        ent_scale_loss = self._cal_scale_loss(self.ent_embedding)
        rel_scale_loss = self._cal_scale_loss(self.rel_embedding)
        return margin_based_loss
    
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
