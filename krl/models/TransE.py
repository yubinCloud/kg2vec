
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ..base_model import XTransEModel, ModelMain
from ..config import TransHyperParam, TrainConf
from ..dataset import KRLDatasetDict
from ..negative_sampler import BernNegSampler
from .. import utils
from ..trainer import TransETrainer
from ..metric import MetricEnum
from ..evaluator import RankEvaluator
from .. import storage
from ..metric_fomatter import StringFormatter
from ..serializer import FileSerializer
from ..lit_model import XTransELitModel


class TransEHyperParam(TransHyperParam):
    """Hyper-paramters of TransE
    """
    pass


class TransE(XTransEModel):
    def __init__(self,
                 ent_num: int,
                 rel_num: int,
                 hyper_params: TransEHyperParam
    ):
        super().__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.norm = hyper_params.norm
        self.embed_dim = hyper_params.embed_dim
        self.margin = hyper_params.margin
    
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

        self.dist_fn = nn.PairwiseDistance(p=self.norm) # the function for calculating the distance 
        self.criterion = nn.MarginRankingLoss(margin=self.margin)
    
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
        r_embs = self.rel_embedding(rels)
        t_embs = self.ent_embedding(tails)
        return h_embs, r_embs, t_embs
    
    def _distance(self, triples):
        """计算一个 batch 的三元组的 distance

        :param triples: 一个 batch 的 triple，size: [batch, 3]
        :return: size: [batch,]
        """
        h_embs, r_embs, t_embs = self.embed(triples)
        return self.dist_fn(h_embs + r_embs, t_embs)
        
    def loss(self, pos_distances, neg_distances):
        """Calculate the loss

        :param pos_distances: [batch, ]
        :param neg_distances: [batch, ]
        :return: loss
        """
        ones = torch.tensor([-1], dtype=torch.long, device=pos_distances.device)
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


class TransEMain(ModelMain):
    def __init__(
        self,
        dataset: KRLDatasetDict,
        train_conf: TrainConf,
        hyper_params: TransEHyperParam,
        device: torch.device
    ) -> None:
        super().__init__()
        self.datasets = dataset
        self.dataset_conf = dataset.dataset_conf
        self.train_conf = train_conf
        self.params = hyper_params
        self.device = device
    
    def __call__(self):
        # create mapping
        entity2id = self.datasets.entity2id
        rel2id = self.datasets.rel2id
        ent_num = len(entity2id)
        rel_num = len(rel2id)
        
        # create dataset and dataloader
        train_dataset = self.datasets.train
        train_dataloader = DataLoader(train_dataset, self.params.batch_size)
        valid_dataset = self.datasets.valid
        valid_dataloader = DataLoader(valid_dataset, self.params.batch_size)
    
        # create negative-sampler
        neg_sampler = BernNegSampler(train_dataset, self.device)

        # create model
        model = TransE(ent_num, rel_num, self.device, self.params)
        model = model.to(self.device)
        
        # create optimizer
        optimizer = utils.create_optimizer(self.params.optimizer, model, self.params.learning_rate)
    
        # create trainer
        trainer = TransETrainer(
            model=model,
            train_conf=self.train_conf,
            params=self.params,
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
        test_dataset = self.datasets.test
        test_dataloder = DataLoader(test_dataset, self.params.valid_batch_size)
        # run inference on test-dataset
        metric = trainer.run_inference(test_dataloder, ent_num, evaluator)
    
        # choice metric formatter
        metric_formatter = StringFormatter()
    
        # choice the way of serialize
        serilizer = FileSerializer(self.train_conf, self.dataset_conf)
        # serialize the metric
        serilizer.serialize(metric, metric_formatter)


class TransELitMain(ModelMain):
    def __init__(
        self,
        dataset: KRLDatasetDict,
        train_conf: TrainConf,
        hyper_params: TransEHyperParam,
        device: torch.device
    ) -> None:
        super().__init__()
        self.datasets = dataset
        self.dataset_conf = dataset.dataset_conf
        self.train_conf = train_conf
        self.params = hyper_params
        self.device = device
    
    def __call__(self):
        # create mapping
        entity2id = self.datasets.entity2id
        rel2id = self.datasets.rel2id
        ent_num = len(entity2id)
        rel_num = len(rel2id)
        
        # create dataset and dataloader
        train_dataset = self.datasets.train
        train_dataloader = DataLoader(train_dataset, self.params.batch_size)
        valid_dataset = self.datasets.valid
        valid_dataloader = DataLoader(valid_dataset, self.params.batch_size)
    
        # create negative-sampler
        neg_sampler = BernNegSampler(train_dataset, self.device)

        # create model
        model = TransE(ent_num, rel_num, self.params)
        model = model.to(self.device)
        model = XTransELitModel(model, self.datasets, neg_sampler, self.params)
        model = model.to(self.device)
        
        # create trainer
        trainer = pl.Trainer(max_epochs=self.params.epoch_size, gpus="0,")
        trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
    