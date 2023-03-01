import typer
from pathlib import Path

from ..config import TrainConf
from ..models.TransE import TransEHyperParam, TransELitMain
from ..dataset import load_krl_dataset


app = typer.Typer()


@app.command(name='train')
def train_transe(
    dataset_name: str = typer.Option('FB15k'),
    batch_size: int = typer.Option(256),
    valid_batch_size: int = typer.Option(16),
    valid_freq: int = typer.Option(5),
    lr: float = typer.Option(0.01),
    optimizer: str = typer.Option('adam'),
    epoch_size: int = typer.Option(500),
    embed_dim: int = typer.Option(100),
    norm: int = typer.Option(1),
    margin: float = typer.Option(2.0),
    logs_dir: Path = typer.Option('lightning_logs/'),
    early_stoping_patience: int = typer.Option(5)
):
    dataset_dict = load_krl_dataset(dataset_name)
    train_conf = TrainConf(
        logs_dir=logs_dir
    )
    hyper_params = TransEHyperParam(
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        epoch_size=epoch_size,
        early_stoping_patience=early_stoping_patience,
        embed_dim=embed_dim,
        norm=norm,
        margin=margin,
        valid_freq=valid_freq
    )
    
    main = TransELitMain(dataset_dict, train_conf, hyper_params)
    
    main()