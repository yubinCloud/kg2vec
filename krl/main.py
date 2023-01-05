from app import app
from typer.testing import CliRunner


if __name__ == '__main__':
    
    runner = CliRunner()
    
    args = [
        'TransR' ,'train',
        '--base-dir', '/root/yubin/dataset/KRL/master/FB15k',
        '--dataset-name', 'FB15k',
        '--batch-size', '4800',
        '--ckpt-path', '/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transr_fb15k.ckpt',
        '--metric-result-path', '/root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transr_fb15k_metrics.txt'
    ]
    
    result = runner.invoke(app, args)
    print(result.stdout)