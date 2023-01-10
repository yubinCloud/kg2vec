"""
Reference:

- https://yubincloud.github.io/notebook-paper/KG/KRL/1101.RESCAL-and-extensions.html
- https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/RESCAL.py
- https://github.com/nju-websoft/muKG/blob/main/src/torch/kge_models/RESCAL.py
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pydantic import Field
from typing import Literal

from base_model import KRLModel, ModelMain
from config import HyperParam, DatasetConf, TrainConf
from dataset import create_mapping, KRLDataset
from negative_sampler import BernNegSampler
import utils
from trainer import RescalTrainer
from metric import MetricEnum
from evaluator import KRLEvaluator
import storage
from metric_fomatter import StringFormatter
from serializer import FileSerializer


class RescalHyperParam(HyperParam):
    """Hyper-parameters of RESCAL
    """
    embed_dim: int
    alpha: float = Field(0.001, title='regularization parameter')
    regul_type: Literal['F2', 'N3'] = Field('F2', title='regularization type')


class RESCAL(KRLModel):
    def __init__(
        self,
        ent_num: int,
        rel_num: int,
        device: torch.device,
        hyper_params: RescalHyperParam,
    ):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.device = device
        self.embed_dim = hyper_params.embed_dim
        self.alpha = hyper_params.alpha
        self.regul_type = hyper_params.regul_type.upper()
        
        # initialize entity embedding
        self.ent_embedding = nn.Embedding(self.ent_num, self.embed_dim)
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, 2, 1)  # 在许多实现中，这一行可以去掉
        
        # initialize relation embedding
        self.rel_embedding = nn.Embedding(self.rel_num, self.embed_dim * self.embed_dim)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, 2, 1)
    
        self.criterion = nn.MSELoss()
    
    def embed(self, triples):
        """Get the embeddings of a batch of triples

        :param triples: _description_
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
        return h_embs, r_embs, t_embs
        
    
    def _scoring(self, h_embs, r_embs, t_embs):
        """计算一个 batch 的三元组的 scores
        score 越大越好，正例接近 1，负例接近 0
        This score can also be regard as the `pred`
        
        :param h_embs: heads embedding，size: [batch, embed]
        :param r_embs: rels embedding，size: [batch, embed * embed]
        :return: size: [batch,]
        """
        # calcate scores
        r_embs = r_embs.view(-1, self.embed_dim, self.embed_dim)  # [batch, emb, emb]
        t_embs = t_embs.view(-1, self.embed_dim, 1)  # [batch, emb, 1]
        
        tr = torch.matmul(r_embs, t_embs)  # [batch, emb, 1]
        tr = tr.view(-1, self.embed_dim)  # [batch, emb]
        
        return torch.sum(h_embs * tr, dim=1)

    def _get_reg(self, h_embs, r_embs, t_embs):
        """Calculate regularization term

        :param h_embs: heads embedding, size: [batch, embed]
        :param r_embs: rels embedding
        :param t_embs: tails embeddings
        :return: _description_
        """
        if self.regul_type == 'F2':
            regul = (torch.mean(h_embs ** 2) + torch.mean(t_embs ** 2) + torch.mean(r_embs ** 2)) / 3
        else:
            regul = torch.mean(torch.sum(h_embs ** 3, -1) + torch.sum(r_embs ** 3, -1) + torch.sum(t_embs ** 3, -1))
        return regul
    
    def loss(self, triples: torch.Tensor, labels: torch.Tensor):
        """Calculate the loss

        :param triples: a batch of triples. size: [batch, 3]
        :param labels: the label of each triple, label = 1 if the triple is positive, label = 0 if the triple is negative. size: [batch,]
        """
        assert triples.shape[1] == 3
        assert triples.shape[0] == labels.shape[0]
        
        h_embs, r_embs, t_embs = self.embed(triples)
        
        scores = self._scoring(h_embs, r_embs, t_embs)
        regul = self._get_reg(h_embs, r_embs, t_embs)
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
        h_embs, r_embs, t_embs = self.embed(triples)
        return -self._scoring(h_embs, r_embs, t_embs)



class RescalMain(ModelMain):
    def __init__(
        self,
        dataset_conf: DatasetConf,
        train_conf: TrainConf,
        hyper_params: RescalHyperParam,
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
        train_dataset, train_dataloader, valid_dataset, valid_dataloader = utils.create_dataloader(self.dataset_conf, self.hyper_params, entity2id, rel2id)
    
        # create negative-sampler
        neg_sampler = BernNegSampler(train_dataset, self.device)

        # create model
        model = RESCAL(ent_num, rel_num, self.device, self.hyper_params)
        model = model.to(self.device)
        
        # create optimizer
        optimizer = utils.create_optimizer(self.hyper_params.optimizer, model, self.hyper_params.learning_rate)
    
        # create trainer
        trainer = RescalTrainer(
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
        evaluator = KRLEvaluator(self.device, metrics)
    
        # Testing the best checkpoint on test dataset
        # load best model
        ckpt = storage.load_checkpoint(self.train_conf)
        model.load_state_dict(ckpt.model_state_dict)
        model = model.to(self.device)
        # create test-dataset
        test_dataset = KRLDataset(self.dataset_conf, 'test', entity2id, rel2id)
        test_dataloder = DataLoader(test_dataset, self.hyper_params.valid_batch_size)
        # run inference on test-dataset
        metric = trainer.run_inference(test_dataloder, ent_num, evaluator)
    
        # choice metric formatter
        metric_formatter = StringFormatter()
    
        # choice the way of serialize
        serilizer = FileSerializer(self.train_conf, self.dataset_conf)
        # serialize the metric
        serilizer.serialize(metric, metric_formatter)
