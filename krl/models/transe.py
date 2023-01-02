
import torch
import torch.nn as nn

from base_model import KRLModel
from config import TransHyperParam


class TransEHyperParam(TransHyperParam):
    """Hyper-paramters of TransE
    """
    pass


class TransE(KRLModel):
    def __init__(self,
                 ent_num: int,
                 rel_num: int,
                 device: torch.device,
                 norm: int,
                 embed_dim: int,
                 margin: float):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.norm = norm
        self.embed_dim = embed_dim
        self.margin = margin
    
        # 初始化 ent_embedding，按照原论文的方法来初始化
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        #uniform_range = 6 / np.sqrt(self.embed_dim)
        #self.ent_embedding.weight.data.uniform_(-uniform_range, uniform_range)
        
        # 初始化 rel_embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        #uniform_range = 6 / np.sqrt(self.embed_dim)
        #self.rel_embedding.weight.data.uniform_(-uniform_range, uniform_range)

        self.criterion = nn.MarginRankingLoss(margin=self.margin)
    
    def _distance(self, triples):
        """计算一个 batch 的三元组的 distance

        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        h_embs = self.ent_embedding(heads)  # h_embs: [batch, embed_dim]
        r_embs = self.rel_embedding(rels)
        t_embs = self.ent_embedding(tails)
        dist = h_embs + r_embs - t_embs  # [batch, embed_dim]
        return torch.norm(dist, p=self.norm, dim=1)
        
    def loss(self, pos_distances, neg_distances):
        """计算 TransE 训练的损失

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: loss
        """
        ones = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(pos_distances, neg_distances, ones)
    
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
