#!/bin/bash

## Pre-training gpt2-medium
python train_gpt2.py \
    --model gpt2-medium \
    --optimizer AdamW \
    --layers-per-ipu 1 7 8 8 \
    --matmul-proportion 0.2 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 1 \
    --gradient-accumulation 512 \
    --batches-per-step 4 \
    --batch-size 4 \
    --embedding-serialization-factor 6 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'tfrecord' \
    --tfrecord-path ./data/tfrecords/*.tfrecord \
    --epochs 3 \
    --use-wandb \
    --save-model-path './checkpoints/gpt2_medium'

