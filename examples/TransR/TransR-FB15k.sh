#!/bin/bash

cd ..  # /examples/
cd ../krl  # /krl
python typer_app.py TransR train --dataset-name "FB15k"\
    --base-dir /root/yubin/dataset/KRL/master/FB15k \
    --batch-size 4800 \
    --valid-batch-size 32 \
    --valid-freq 25 \
    --lr 0.001 \
    --epoch-size 400 \
    --embed-dim 50 \
    --norm 1 \
    --margin 1.0 \
    --c 0.1 \
    --ckpt-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transr_fb15k.ckpt \
    --metric-result-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transr_fb15k_metrics.txt
