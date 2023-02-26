import typer
from pathlib import Path

from ..config import TrainConf
from ..models.TransE import TransEHyperParam, TransEMain, TransELitMain
from .. import utils
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
    epoch_size: int = typer.Option(200),
    embed_dim: int = typer.Option(50),
    norm: int = typer.Option(1),
    margin: float = typer.Option(2.0),
    ckpt_path: Path = typer.Option('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k.ckpt'),
    metric_result_path: Path = typer.Option('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k_metrics.txt')
):
    dataset_dict = load_krl_dataset(dataset_name)
    train_conf = TrainConf(
        checkpoint_path=ckpt_path.absolute().as_posix(),
        metric_result_path=metric_result_path.absolute().as_posix()
    )
    hyper_params = TransEHyperParam(
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        epoch_size=epoch_size,
        embed_dim=embed_dim,
        norm=norm,
        margin=margin,
        valid_freq=valid_freq
    )
    device = utils.get_device()
    
    main = TransEMain(dataset_dict, train_conf, hyper_params, device)
    
    main()


@app.command(name='pl-train')
def train_transe(
    dataset_name: str = typer.Option('FB15k'),
    batch_size: int = typer.Option(256),
    valid_batch_size: int = typer.Option(16),
    valid_freq: int = typer.Option(5),
    lr: float = typer.Option(0.01),
    optimizer: str = typer.Option('adam'),
    epoch_size: int = typer.Option(200),
    embed_dim: int = typer.Option(50),
    norm: int = typer.Option(1),
    margin: float = typer.Option(2.0),
    ckpt_path: Path = typer.Option('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k.ckpt'),
    metric_result_path: Path = typer.Option('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k_metrics.txt')
):
    dataset_dict = load_krl_dataset(dataset_name)
    train_conf = TrainConf(
        checkpoint_path=ckpt_path.absolute().as_posix(),
        metric_result_path=metric_result_path.absolute().as_posix()
    )
    hyper_params = TransEHyperParam(
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        epoch_size=epoch_size,
        embed_dim=embed_dim,
        norm=norm,
        margin=margin,
        valid_freq=valid_freq
    )
    
    main = TransELitMain(dataset_dict, train_conf, hyper_params)
    
    main()