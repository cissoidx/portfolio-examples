#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"./logs/gpt2_report8-2-1","debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true","profiler.perExecutionStreamCopyCycles":"true"}'
# export POPLAR_ENGINE_OPTIONS='{"profiler.includeFlopEstimates": "true","autoReport.all":"true","debug.allowOutOfMemory": "true", "autoReport.executionProfileProgramRunCount":"1", "autoReport.directory":"./logs/small_training_new"}'
export WANDB_BASE_URL=https://wandb.sourcevertex.net

# Pre-training gpt2-small
poprun -vv --num-instances=1 --num-replicas=4 \
        --ipus-per-replica=4 \
        --vipu-partition=p64_a2 \
        --update-partition=no \
        --remove-partition=no \
        --reset-partition=no \
        --print-topology=yes \
        --mpi-global-args=" --tag-output" \
python train_gpt2.py \
   --model gpt2-medium \
   --train_path './data/wikicorpus_en_one_article_per_line.pkl' \
   --stride 128 \
   --layers_per_ipu 1 7 8 8 \
   --matmul_proportion 0.6 0.6 0.6 0.6 \
   --replication_factor 4 \
   --ipus_per_replica 4 \
   --gradient_accumulation 512 \
   --batches_per_step 8 \
   --batch_size 4 \
   --epochs 30 \
   --lr_schedule 'linear' \
   --embedding_serialization_factor 2 \
   --recompute_checkpoint_every_layer True \
   --enable_half_partials True \
   --loss_scaling 50000 \
   --use_wandb \
   --executable_cache_dir '/localdata/chaon/tmp' \
   --save_model_path './checkpoints/gpt2_medium' 2>&1 | tee logs/gpt2_medium_30522_$time.log