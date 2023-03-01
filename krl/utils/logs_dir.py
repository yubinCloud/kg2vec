from datetime import datetime

from ..config import TrainConf, DatasetConf
from ..base_model import KRLModel


def create_logs_dir(
    model: KRLModel,
    train_conf: TrainConf,
    dataset_conf: DatasetConf
) -> str:  
    dir = train_conf.logs_dir / model.__class__.__name__ / dataset_conf.dataset_name / datetime.now().strftime(r'%y%m%d-%H%M%S')
    dir.mkdir(parents=True, exist_ok=True)
    return str(dir)
