#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")

#export POPLAR_ENGINE_OPTIONS='{"profiler.includeFlopEstimates": "true","autoReport.all":"true","debug.allowOutOfMemory": "true", "autoReport.executionProfileProgramRunCount":"1", "autoReport.directory":"./logs/small_mlp_bs16"}'

## Pre-training gpt2-medium
python train_gpt2.py \
    --model gpt2-medium \
    --optimizer AdamW \
    --lr-schedule 'linear' \
    --layers-per-ipu 1 7 8 8 \
    --matmul-proportion 0.2 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --replication-factor 4 \
    --gradient-accumulation 512 \
    --batches-per-step 4 \
    --batch-size 4 \
    --embedding-serialization-factor 6 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path 'data/wikicorpus_en_one_article_per_line.pkl' \
    --epochs 3 \
    --save-model-path './checkpoints/gpt2_medium'

