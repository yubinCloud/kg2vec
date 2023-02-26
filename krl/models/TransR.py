"""
Reference:

- https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransR.py
- https://github.com/zqhead/TransR
- https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/models/pairwise.py
- https://github.com/nju-websoft/muKG/blob/main/src/torch/kge_models/TransR.py

Note: Although the TransE can be run, I don't know why it have very low hits@10 metric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base_model import TransXBaseModel, ModelMain
from ..config import HyperParam, LocalDatasetConf, TrainConf
from ..dataset import create_mapping, LocalKRLDataset
from ..negative_sampler import BernNegSampler
from .. import utils
from ..trainer import TransETrainer
from ..metric import MetricEnum
from ..evaluator import RankEvaluator
from .. import storage
from ..metric_fomatter import StringFormatter
from ..serializer import FileSerializer

class TransRHyperParam(HyperParam):
    """Hyper-parameters of TransR
    """
    embed_dim: int
    norm: int
    margin: float
    C: float
    

class TransR(TransXBaseModel):
    def __init__(self,
                 ent_num: int,
                 rel_num: int,
                 device: torch.device,
                 hyper_params: TransRHyperParam
    ):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.norm = hyper_params.norm
        self.embed_dim = hyper_params.embed_dim
        self.C = hyper_params.C
        
        self.margin = hyper_params.margin
        self.epsilon = 2.0
        self.embedding_range = (self.margin + self.epsilon) / self.embed_dim

        # 初始化 ent_embedding，按照原论文的方法来初始化
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        nn.init.uniform_(
            tensor=self.ent_embedding.weight.data,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        # torch.nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        
        # 初始化 rel_embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim)
        nn.init.uniform_(
            tensor=self.rel_embedding.weight.data,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        # torch.nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        
        # initialize trasfer matrix
        self.transfer_matrix = nn.Embedding(self.rel_num, self.embed_dim * self.embed_dim)
        nn.init.uniform_(
            tensor=self.transfer_matrix.weight.data,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        # nn.init.xavier_uniform_(self.tranfer_matrix.weight.data)

        self.dist_fn = nn.PairwiseDistance(p=self.norm) # the function for calculating the distance 
        self.margin_loss_fn = nn.MarginRankingLoss(margin=self.margin)
    
    def _transfer(self, ent_embs: torch.Tensor, rels_tranfer: torch.Tensor):
        """Tranfer the entity space into the relation-specfic space

        :param ent_embs: [batch, ent_dim]
        :param rels_tranfer: [batch, ent_dim * rel_dim]
        """
        assert ent_embs.size(0) == rels_tranfer.size(0)
        assert ent_embs.size(1) == self.embed_dim
        assert rels_tranfer.size(1) == self.embed_dim * self.embed_dim
            
        rels_tranfer = rels_tranfer.view(-1, self.embed_dim, self.embed_dim)    # [batch, ent_dim, rel_dim]
        ent_embs = ent_embs.view(-1, 1, self.embed_dim)   # [batch, 1, ent_dim]
            
        ent_proj = torch.matmul(ent_embs, rels_tranfer)   # [batch, 1, rel_dim]
        return ent_proj.view(-1, self.embed_dim)   # [batch, rel_dim]
    
    def embed(self, triples):
        """get the embedding of triples

        :param triples: [heads, rels, tails]
        :return: embedding of triples.
        """
        assert triples.shape[1] == 3
        heads = triples[:, 0]
        rels = triples[:, 1]
        tails = triples[:, 2]
        # id -> embedding
        h_embs = self.ent_embedding(heads)  # h_embs: [batch, embed_dim]
        r_embs = self.rel_embedding(rels)
        t_embs = self.ent_embedding(tails)
        rels_transfer = self.transfer_matrix(rels)
        # tranfer the entity embedding from entity space into relation-specific space
        h_embs = self._transfer(h_embs, rels_transfer)
        t_embs = self._transfer(t_embs, rels_transfer)
        return h_embs, r_embs, t_embs
    
    def _distance(self, triples):
        """计算一个 batch 的三元组的 distance

        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
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
        return margin_based_loss + self.C * ((ent_scale_loss + rel_scale_loss) / (self.ent_num + self.rel_num))
    
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


class TransRMain(ModelMain):
    
    def __init__(
        self,
        dataset_conf: LocalDatasetConf,
        train_conf: TrainConf,
        hyper_params: TransRHyperParam,
        device: torch.device
    ) -> None:
        super().__init__()
        self.dataset_conf = dataset_conf
        self.train_conf = train_conf
        self.hyper_params = hyper_params
        self.device = device
    
    def __call__(self):
        # create mapping
        entity2id, rel2id = create_mapping(self.dataset_conf)
        ent_num = len(entity2id)
        rel_num = len(rel2id)
        
        # create dataset and dataloader
        train_dataset, train_dataloader, valid_dataset, valid_dataloader = utils.create_local_dataloader(self.dataset_conf, self.hyper_params, entity2id, rel2id)
    
        # create negative-sampler
        neg_sampler = BernNegSampler(train_dataset, self.device)

        # create model
        model = TransR(ent_num, rel_num, self.device, self.hyper_params)
        model = model.to(self.device)
        
        # create optimizer
        optimizer = utils.create_optimizer(self.hyper_params.optimizer, model, self.hyper_params.learning_rate)
    
        # create trainer
        trainer = TransETrainer(
            model=model,
            train_conf=self.train_conf,
            params=self.hyper_params,
            dataset_conf=self.dataset_conf,
            entity2id=entity2id,
            rel2id=rel2id,
            device=self.device,
            train_dataloder=train_dataloader,
            valid_dataloder=valid_dataloader,
            train_neg_sampler=neg_sampler,
            valid_neg_sampler=neg_sampler,
            optimzer=optimizer
        )
    
        # training process
        trainer.run_training()
    
        # create evaluator
        metrics = [
            MetricEnum.MRR,
            MetricEnum.HITS_AT_1,
            MetricEnum.HITS_AT_3,
            MetricEnum.HITS_AT_10
        ]
        evaluator = RankEvaluator(self.device, metrics)
    
        # Testing the best checkpoint on test dataset
        # load best model
        ckpt = storage.load_checkpoint(self.train_conf)
        model.load_state_dict(ckpt.model_state_dict)
        model = model.to(self.device)
        # create test-dataset
        test_dataset = LocalKRLDataset(self.dataset_conf, 'test', entity2id, rel2id)
        test_dataloder = DataLoader(test_dataset, self.hyper_params.valid_batch_size)
        # run inference on test-dataset
        metric = trainer.run_inference(test_dataloder, ent_num, evaluator)
    
        # choice metric formatter
        metric_formatter = StringFormatter()
    
        # choice the way of serialize
        serilizer = FileSerializer(self.train_conf, self.dataset_conf)
        # serialize the metric
        serilizer.serialize(metric, metric_formatter)
