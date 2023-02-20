from pathlib import Path
import typer

from ..config import TrainConf
from ..models.RESCAL import RescalHyperParam, RescalMain
from .. import utils
from ..dataset import load_krl_dataset


app = typer.Typer()


@app.command(name='train')
def train_rescal(
    dataset_name: str = typer.Option('FB15k'),
    batch_size: int = typer.Option(256),
    valid_batch_size: int = typer.Option(16),
    valid_freq: int = typer.Option(5),
    lr: float = typer.Option(0.01),
    epoch_size: int = typer.Option(200),
    optimizer: str = typer.Option('adam'),
    embed_dim: int = typer.Option(50),
    alpha: float = typer.Option(0.001, help='regularization parameter'),
    regul_type: str = typer.Option('F2', help='regularization type, F2 or N3', case_sensitive=False),
    ckpt_path: Path = typer.Option(Path('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/rescal_fb15k.ckpt')),
    metric_result_path: Path = typer.Option(Path('/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/rescal_fb15k_metrics.txt'))
):
    dataset_dict = load_krl_dataset(dataset_name)
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
    
    main = RescalMain(dataset_dict, train_conf, hyper_params, device)
    
    main()
