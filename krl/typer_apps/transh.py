from pathlib import Path
import typer

from config import DatasetConf, TrainConf
from models.transh import TransHHyperParam, TransHMain
import utils


app = typer.Typer()


@app.command(name='train')
def train_tranh(
    dataset_name: str = typer.Option(...),
    base_dir: Path = typer.Option(...),
    batch_size: int = typer.Option(128),
    valid_batch_size: int = typer.Option(64),
    valid_freq: int = typer.Option(5),
    lr: float = typer.Option(0.001),
    epoch_size: int = typer.Option(500),
    optimizer: str = typer.Option('adam'),
    embed_dim: int = typer.Option(50),
    norm: int = typer.Option(2),
    margin: float = typer.Option(1.0),
    C: float = typer.Option(1.0, help='a hyper-parameter weighting the importance of soft constraints.'),
    eps: float = typer.Option(1e-3, help='the $\episilon$ in loss function'),
    ckpt_path: Path = typer.Option(...),
    metric_result_path: Path = typer.Option(...)
):
    if not base_dir.exists():
        print("base_dir doesn't exists.")
        raise typer.Exit()
    # initialize all configurations
    dataset_conf = DatasetConf(
        dataset_name=dataset_name,
        base_dir=base_dir.absolute().as_posix()
    )
    train_conf = TrainConf(
        checkpoint_path=ckpt_path.absolute().as_posix(),
        metric_result_path=metric_result_path.absolute().as_posix()
    )
    hyper_params = TransHHyperParam(
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        epoch_size=epoch_size,
        embed_dim=embed_dim,
        norm=norm,
        margin=margin,
        valid_freq=valid_freq,
        C=C,
        eps=eps
    )
    device = utils.get_device()
    
    main = TransHMain(dataset_conf, train_conf, hyper_params, device)
    
    main()
