from pathlib import Path
import typer

from config import DatasetConf, TrainConf
from models.RESCAL import RescalHyperParam, RescalMain
import utils


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
        base_dir=base_dir
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
    device = utils.get_device()
    
    main = RescalMain(dataset_conf, train_conf, hyper_params, device)
    
    main()
