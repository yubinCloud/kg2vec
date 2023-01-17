import typer
from pathlib import Path

from config import DatasetConf, TrainConf
from models.TransE import TransEHyperParam, TransEMain
import utils


app = typer.Typer()

@app.command(name='train')
def train_transe(
    dataset_name: str = typer.Option(...),
    base_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(...),
    valid_batch_size: int = typer.Option(...),
    valid_freq: int = typer.Option(...),
    lr: float = typer.Option(...),
    optimizer: str = typer.Option('adam'),
    epoch_size: int = typer.Option(...),
    embed_dim: int = typer.Option(...),
    norm: int = typer.Option(...),
    margin: float = typer.Option(...),
    ckpt_path: Path = typer.Option(...),
    metric_result_path: Path = typer.Option(...)
):
    if not base_dir.exists():
        print("base_dir doesn't exists.")
        raise typer.Exit()
    dataset_conf = DatasetConf(
        dataset_name=dataset_name,
        base_dir=base_dir
    )
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
    
    main = TransEMain(dataset_conf, train_conf, hyper_params, device)
    
    main()