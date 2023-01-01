
from pydantic import BaseModel
import torch

from config import TrainConf, HyperParam
from base_model import KRLModel

class CheckpointFormat(BaseModel):
    model_state_dict: dict
    optim_state_dict: dict
    epoch_id: int
    best_score: float
    hyper_params: dict


def save_checkpoint(model: KRLModel,
                    optimzer: torch.optim.Optimizer,
                    epoch_id: int,
                    best_score: float,
                    hyper_params: HyperParam,
                    train_conf: TrainConf):
    ckpt = CheckpointFormat(
        model_state_dict=model.state_dict(),
        optim_state_dict=optimzer.state_dict(),
        epoch_id=epoch_id,
        best_score=best_score,
        hyper_params=hyper_params.dict()
    )
    torch.save(ckpt.dict(), train_conf.checkpoint_path)


def load_checkpoint(train_conf: TrainConf) -> CheckpointFormat:
    ckpt = torch.load(train_conf.checkpoint_path)
    return CheckpointFormat.parse_obj(ckpt)
