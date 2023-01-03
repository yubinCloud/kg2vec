"""
KRLTrainer for training and testing models.
"""
from typing import Mapping
import torch
from torch.utils.data import DataLoader
from typing import Tuple
from rich.progress import track
from abc import ABC, abstractclassmethod

from base_model import KRLModel
from config import TrainConf, HyperParam, DatasetConf
from negative_sampler import NegativeSampler
import metric, storage




class KRLTrainer(ABC):
    def __init__(self,
                 model: KRLModel,
                 train_conf: TrainConf,
                 params: HyperParam,
                 dataset_conf: DatasetConf,
                 entity2id: Mapping[str ,int],
                 rel2id: Mapping[str, int],
                 device: torch.device,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 train_neg_sampler: NegativeSampler,
                 valid_neg_sampler: NegativeSampler,
                 optimzer: torch.optim.Optimizer
                ) -> None:
       self.model = model
       self.train_conf = train_conf
       self.params = params
       self.dataset_conf = dataset_conf
       self.entity2id = entity2id
       self.rel2id = rel2id
       self.device = device
       self.train_dataloader = train_dataloader
       self.valid_dataloader = valid_dataloader
       self.train_neg_sampler = train_neg_sampler
       self.valid_neg_sampler = valid_neg_sampler
       self.optimzer = optimzer
    
    def run_inference(self,
                      dataloder: DataLoader,
                      ent_num: int
                      ) -> Tuple[float, float, float, float]:
        """
        Run the inference process on the KRL model.
        
        Rewrite this function if you need more logic for the training model.
        The implementation here just provides an example of training TransE.
        """
        hits_at_1 = 0.0
        hits_at_3 = 0.0
        hits_at_10 = 0.0
        mrr = 0.0
        examples_count = 0
        
        model = self.model
        device = self.device
        
        # entity_ids = [[0, 1, 2, ..., ent_num]], shape: [1, ent_num]
        entitiy_ids = torch.arange(0, ent_num, device=device).unsqueeze(0)
        for i, batch in enumerate(dataloder):
            # batch: [3, batch_size]
            heads, rels, tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            batch_size = heads.size()[0]
            all_entities = entitiy_ids.repeat(batch_size, 1)  # all_entities: [batch_size, ent_num]
            # heads: [batch_size,] -> [batch_size, 1] -> [batch_size, ent_num]
            heads_expanded = heads.reshape(-1, 1).repeat(1, ent_num)  # _expanded: [batch_size, ent_num]
            rels_expanded = rels.reshape(-1, 1).repeat(1, ent_num)
            tails_expanded = tails.reshape(-1, 1).repeat(1, ent_num)
            # check all possible tails
            triplets = torch.stack([heads_expanded, rels_expanded, all_entities], dim=2).reshape(-1, 3)  # triplets: [batch_size * ent_num, 3]
            tails_predictions = model.predict(triplets).reshape(batch_size, -1)  # tails_prediction: [batch_size, ent_num]
            # check all possible heads
            triplets = torch.stack([all_entities, rels_expanded, tails_expanded], dim=2).reshape(-1, 3)
            heads_predictions = model.predict(triplets).reshape(batch_size, -1)  # heads_prediction: [batch_size, ent_num]

            # Concept preditions
            predictions = torch.cat([tails_predictions, heads_predictions], dim=0)  # predictions: [batch_size * 2, ent_num]
            ground_truth_entity_id = torch.cat([tails.reshape(-1, 1), heads.reshape(-1, 1)], dim=0)  # [batch_size * 2, 1]
            # calculate metrics
            hits_at_1 += metric.cal_hits_at_k(predictions, ground_truth_entity_id, device=device, k=1)
            hits_at_3 += metric.cal_hits_at_k(predictions, ground_truth_entity_id, device=device, k=3)
            hits_at_10 += metric.cal_hits_at_k(predictions, ground_truth_entity_id, device=device, k=10)
            mrr += metric.cal_mrr(predictions, ground_truth_entity_id)
        
            examples_count += predictions.size()[0]
        
        hits_at_1_score = hits_at_1 / examples_count * 100
        hits_at_3_score = hits_at_3 / examples_count * 100
        hits_at_10_score = hits_at_10 / examples_count * 100
        mrr_score = mrr / examples_count * 100
        
        return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score
    
    @abstractclassmethod
    def run_training(self):
        """
        Run the training process on the KRL model.
        
        Rewrite this function if you need more logic for the training model.
        The implementation here just provides an example of training TransE.
        """
        pass


class TransETrainer(KRLTrainer):
    """
    Trainer for training TransE and other similar models.
    """
    def __init__(
        self,
        model: KRLModel,
        train_conf: TrainConf,
        params: HyperParam,
        dataset_conf: DatasetConf,
        entity2id: Mapping[str, int],
        rel2id: Mapping[str, int],
        device: torch.device,
        train_dataloder: DataLoader,
        valid_dataloder: DataLoader,
        train_neg_sampler: NegativeSampler,
        valid_neg_sampler: NegativeSampler,
        optimzer: torch.optim.Optimizer
    ) -> None:
        super().__init__(model, train_conf, params, dataset_conf, entity2id, rel2id, device, train_dataloder, valid_dataloder, train_neg_sampler, valid_neg_sampler, optimzer)
    
    def run_training(self):
        """
        Run the training process on the TransE model and other similar models, for example, TransH.
        
        Rewrite this function if you need more logic for the training model.
        The implementation here just provides an example of training TransE.
        """
        device = self.device
        optimzer = self.optimzer
        model = self.model
        # prepare the tools for tarining
        best_score = 0.0
        # training loop
        for epoch_id in track(range(1, self.params.epoch_size + 1), description='Total...'):
            print("Starting epoch: ", epoch_id)
            loss_sum = 0
            model.train()
            for i, batch in enumerate(self.train_dataloader):
                # get a batch of training data
                pos_heads, pos_rels, pos_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                pos_triples = torch.stack([pos_heads, pos_rels, pos_tails], dim=1)  # pos_triples: [batch_size, 3]
                neg_triples = self.train_neg_sampler.neg_sample(pos_heads, pos_rels, pos_tails)  # neg_triples: [batch_size, 3]
                optimzer.zero_grad()
                # calculte loss
                loss, _, _ = model(pos_triples, neg_triples)
                loss.backward()
                loss_sum += loss.cpu().item()
                # update model
                optimzer.step()
            
            if epoch_id % self.params.valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    ent_num = len(self.entity2id)
                    _, _, hits_at_10, _ = self.run_inference(self.valid_dataloader, ent_num)
                    if hits_at_10 > best_score:
                        best_score = hits_at_10
                        print('best score of valid: ', best_score)
                        storage.save_checkpoint(model, optimzer, epoch_id, best_score, self.params, self.train_conf)


class RescalTrainer(KRLTrainer):
    """
    Trainer for tarining RESCAL and other similar models.
    """
    def __init__(self, model: KRLModel, train_conf: TrainConf, params: HyperParam, dataset_conf: DatasetConf, entity2id: Mapping[str, int], rel2id: Mapping[str, int], device: torch.device, train_dataloder: DataLoader, valid_dataloder: DataLoader, train_neg_sampler: NegativeSampler, valid_neg_sampler: NegativeSampler, optimzer: torch.optim.Optimizer) -> None:
        super().__init__(model, train_conf, params, dataset_conf, entity2id, rel2id, device, train_dataloder, valid_dataloder, train_neg_sampler, valid_neg_sampler, optimzer)
    
    def run_training(self):
        device = self.device
        optimzer = self.optimzer
        model = self.model
        # prepare tools for training
        best_score = 0.0
        # training loop
        for epoch_id in track(range(1, self.params.epoch_size + 1), description='Total...'):
            print("Starting epoch: ", epoch_id)
            loss_sum = 0
            model.train()
            for batch in iter(self.train_dataloader):
                pos_heads, pos_rels, pos_tails = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                pos_triples = torch.stack([pos_heads, pos_rels, pos_tails], dim=1)  # pos_triples: [batch_size, 3]
                neg_triples = self.train_neg_sampler.neg_sample(pos_heads, pos_rels, pos_tails)  # neg_triples: [batch_size, 3]
                triples = torch.cat([pos_triples, neg_triples])
                pos_num = pos_triples.size(0)
                total_num = triples.size(0)
                labels = torch.zeros([total_num], device=device)
                labels[0: pos_num] = 1  # the pos_triple label is equal to 1. 
                shuffle_index = torch.randperm(total_num, device=device)  # index sequence for shuffling data
                triples = triples[shuffle_index]
                labels = labels[shuffle_index]
                # calculate loss
                optimzer.zero_grad()
                loss, _ = model(triples, labels)
                loss.backward()
                loss_sum += loss.cpu().item()
                # update model
                optimzer.step()
            
            if epoch_id % self.params.valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    ent_num = len(self.entity2id)
                    _, _, hits_at_10, _ = self.run_inference(self.valid_dataloader, ent_num)
                    if hits_at_10 > best_score:
                        best_score = hits_at_10
                        print('best score of valid: ', best_score)
                        storage.save_checkpoint(model, optimzer, epoch_id, best_score, self.params, self.train_conf)
