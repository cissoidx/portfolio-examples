#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"./logs/gpt2_report8-2-1","debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true","profiler.perExecutionStreamCopyCycles":"true"}'
# export POPLAR_ENGINE_OPTIONS='{"profiler.includeFlopEstimates": "true","autoReport.all":"true","debug.allowOutOfMemory": "true", "autoReport.executionProfileProgramRunCount":"1", "autoReport.directory":"./logs/small_training_new"}'
#export WANDB_BASE_URL=https://wandb.sourcevertex.net

# Pre-training
poprun -vv --num-instances=1 --num-replicas=4 \
        --ipus-per-replica=4 \
        --vipu-partition=p64 \
        --update-partition=no \
        --remove-partition=no \
        --reset-partition=no \
        --print-topology=yes \
        --mpi-global-args=" --tag-output" \
python train_gpt2.py \
    --model gpt2-medium \
    --optimizer LAMB \
    --layers-per-ipu 1 7 8 8 \
    --matmul-proportion 0.2 0.15 0.15 0.15 \
    --ipus-per-replica 4 \
    --gradient-accumulation 1024 \
    --batches-per-step 16 \
    --batch-size 4 \
    --embedding-serialization-factor 6 \
    --mlp-serialization-factor 1 \
    --recompute-checkpoint-every-layer True \
    --enable-half-partials True \
    --train-path './data/wikicorpus_en_one_article_per_line.pkl' \
    --num-workers 64 \
    --save-model-path './checkpoints/gpt2_medium' 2>&1 | tee logs/gpt2_medium_30522_$time.log