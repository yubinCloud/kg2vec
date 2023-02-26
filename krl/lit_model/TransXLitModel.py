import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import List, Any

from ..base_model import TransXBaseModel
from ..config import HyperParam
from ..negative_sampler import NegativeSampler
from ..metric import HitsAtK, MRR
from ..dataset import KRLDatasetDict
from .. import utils


class TransXLitModel(pl.LightningModule):
    def __init__(
        self,
        model: TransXBaseModel,
        dataset_dict: KRLDatasetDict,
        train_neg_sampler: NegativeSampler,
        hyper_params: HyperParam
    ) -> None:
        super().__init__()
        self.model = model
        self.model_device = next(model.parameters()).device
        self.dataset_dict = dataset_dict
        self.train_neg_sampler = train_neg_sampler
        self.params = hyper_params
        self.val_hits10 = HitsAtK(10)
        self.test_hits1 = HitsAtK(1)
        self.test_hits10 = HitsAtK(10)
        self.test_mrr = MRR()
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor):
        """
        training step
        :param batch: [3, batch_size]
        :param batch_idx: [batch_size]
        """
        pos_heads, pos_rels, pos_tails = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        pos_triples = torch.stack([pos_heads, pos_rels, pos_tails], dim=1)  # pos_triples: [batch_size, 3]
        neg_triples = self.train_neg_sampler.neg_sample(pos_heads, pos_rels, pos_tails)  # neg_triples: [batch_size, 3]
        # calculte loss
        loss, _, _ = self.model.forward(pos_triples, neg_triples)
        return loss
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor):
        preds, target = self._get_preds_and_target(batch)
        self.val_hits10.update(preds, target)
    
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_hits_at_10 = self.val_hits10.compute()
        self.val_hits10.reset()
        self.log('val_hits_at_10', val_hits_at_10)
    
    def test_step(self, batch: List[torch.Tensor], batch_idx: torch.Tensor):
        preds, target = self._get_preds_and_target(batch)
        self.test_hits1.update(preds, target)
        self.test_hits10.update(preds, target)
        self.test_mrr.update(preds, target)
    
    def test_epoch_end(self, outputs: List[Any]) -> None:
        result = {
            'hits_at_1': self.test_hits1.compute(),
            'hits_at_10': self.test_hits10.compute(),
            'mrr': self.test_mrr.compute()
        }
        
        self.test_hits1.reset()
        self.test_hits10.reset()
        self.test_mrr.reset()
        
        self.log_dict(result)
    
    def configure_optimizers(self):
        optimizer = utils.create_optimizer(self.params.optimizer, self.model, self.params.learning_rate)
        milestones = int(self.params.epoch_size / 2)
        stepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': stepLR
        }
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict.train,
            batch_size=self.params.batch_size
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict.valid,
            batch_size=self.params.valid_batch_size
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_dict.test,
            batch_size=self.params.valid_batch_size
        )
        
    
    def _get_preds_and_target(self, batch: List[torch.Tensor]):
        ent_num = len(self.dataset_dict.meta.entity2id)
        entity_ids = torch.arange(0, ent_num, device=self.device)
        entity_ids.unsqueeze_(0)
        heads, rels, tails = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        batch_size = heads.size(0)
        all_entities = entity_ids.repeat(batch_size, 1)  # all_entities: [batch_size, ent_num]
        # heads: [batch_size,] -> [batch_size, 1] -> [batch_size, ent_num]
        heads_expanded = heads.reshape(-1, 1).repeat(1, ent_num)  # _expanded: [batch_size, ent_num]
        rels_expanded = rels.reshape(-1, 1).repeat(1, ent_num)
        tails_expanded = tails.reshape(-1, 1).repeat(1, ent_num)
        # check all possible tails
        triplets = torch.stack([heads_expanded, rels_expanded, all_entities], dim=2).reshape(-1, 3)  # triplets: [batch_size * ent_num, 3]
        tails_predictions = self.model.predict(triplets).reshape(batch_size, -1)  # tails_prediction: [batch_size, ent_num]
        # check all possible heads
        triplets = torch.stack([all_entities, rels_expanded, tails_expanded], dim=2).reshape(-1, 3)
        heads_predictions = self.model.predict(triplets).reshape(batch_size, -1)  # heads_prediction: [batch_size, ent_num]

        # Concept preditions
        predictions = torch.cat([tails_predictions, heads_predictions], dim=0)  # predictions: [batch_size * 2, ent_num]
        ground_truth_entity_id = torch.cat([tails.reshape(-1, 1), heads.reshape(-1, 1)], dim=0)  # [batch_size * 2, 1]
        
        return predictions, ground_truth_entity_id
