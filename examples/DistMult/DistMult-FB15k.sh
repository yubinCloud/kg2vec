#!/bin/bash

cd ..  # /examples/
cd ../krl  # /krl
python app.py DistMult train --dataset-name "FB15k"\
    --base-dir /root/yubin/dataset/KRL/master/FB15k \
    --batch-size 128 \
    --valid-batch-size 32 \
    --valid-freq 3 \
    --lr 0.001 \
    --epoch-size 500 \
    --embed-dim 50 \
    --alpha 0.001 \
    --regul-type F2 \
    --ckpt-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/distmult_fb15k.ckpt \
    --metric-result-path /root/sharespace/yubin/papers/KRL/scratch/TransX/tmp/distmult_fb15k_metrics.txt
