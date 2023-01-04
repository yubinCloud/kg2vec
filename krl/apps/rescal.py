from torch.utils.data import DataLoader
from pathlib import Path
import typer

from config import DatasetConf, TrainConf
from models.rescal import RESCAL, RescalHyperParam
from dataset import create_mapping, KRLDataset
from trainer import RescalTrainer
from negative_sampler import RandomNegativeSampler
import storage
import utils
from evaluator import KRLEvaluator
from metric_fomatter import StringFormatter
from serializer import FileSerializer
from metric import MetricEnum


app = typer.Typer()


@app.command(name='train')
def train_rescal(
    dataset_name: str = typer.Option(...),
    base_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(...),
    valid_batch_size: int = typer.Option(...),
    valid_freq: int = typer.Option(...),
    lr: float = typer.Option(...),
    epoch_size: int = typer.Option(...),
    optimizer: str = typer.Option('adam'),
    embed_dim: int = typer.Option(...),
    alpha: float = typer.Option(0.001, help='regularization parameter'),
    regul_type: str = typer.Option('F2', help='regularization type, F2 or N3', case_sensitive=False),
    ckpt_path: Path = typer.Option(...),
    metric_result_path: Path = typer.Option(...)
):
    if not base_dir.exists():
        print("base_dir doesn't exists.")
        raise typer.Exit()
    dataset_conf = DatasetConf(
        dataset_name=dataset_name,
        base_dir=base_dir.absolute().as_posix()
    )
    train_conf = TrainConf(
        checkpoint_path=ckpt_path.absolute().as_posix(),
        metric_result_path=metric_result_path.absolute().as_posix()
    )
    hyper_params = RescalHyperParam(
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        epoch_size=epoch_size,
        embed_dim=embed_dim,
        valid_freq=valid_freq,
        alpha=alpha,
        regul_type=regul_type
    )
    # create mapping
    entity2id, rel2id = create_mapping(dataset_conf)
    device = utils.get_device()
    ent_num = len(entity2id)
    rel_num = len(rel2id)
    
    # create dataset and dataloader
    train_dataset, train_dataloader, valid_dataset, valid_dataloader = utils.create_dataloader(dataset_conf, hyper_params, entity2id, rel2id)
    
    # create negative-sampler
    neg_sampler = RandomNegativeSampler(train_dataset, device)
    
    # create model
    model = RESCAL(ent_num, rel_num, device, embed_dim, alpha, regul_type)
    model = model.to(device)
    
    # create optimizer
    optimizer = utils.create_optimizer(optimizer, model, hyper_params.learning_rate)
    
    # create trainer
    trainer = RescalTrainer(
        model=model,
        train_conf=train_conf,
        params=hyper_params,
        dataset_conf=dataset_conf,
        entity2id=entity2id,
        rel2id=rel2id,
        device=device,
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
    evaluator = KRLEvaluator(device, metrics)
    
    # Testing the best checkpoint on test dataset
    # load best model
    ckpt = storage.load_checkpoint(train_conf)
    model.load_state_dict(ckpt.model_state_dict)
    model = model.to(device)
    # create test-dataset
    test_dataset = KRLDataset(dataset_conf, 'test', entity2id, rel2id)
    test_dataloder = DataLoader(test_dataset, hyper_params.valid_batch_size)
    # run inference on test-dataset
    metric = trainer.run_inference(test_dataloder, ent_num, evaluator)
    
    # choice metric formatter
    metric_formatter = StringFormatter()
    
    # choice the way of serialize
    serilizer = FileSerializer(train_conf, dataset_conf)
    # serialize the metric
    serilizer.serialize(metric, metric_formatter)
