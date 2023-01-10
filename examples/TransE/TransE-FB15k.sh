#!/bin/bash

cd .. # /examples
cd ../krl  # /krl
python typer_app.py TransE train --dataset-name "FB15k"\
    --base-dir /root/yubin/dataset/KRL/master/FB15k \
    --batch-size 128 \
    --valid-batch-size 64 \
    --valid-freq 5 \
    --lr 0.001 \
    --epoch-size 500 \
    --embed-dim 50 \
    --norm 1 \
    --margin 2.0 \
    --ckpt-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k.ckpt \
    --metric-result-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/transe_fb15k_metrics.txt
