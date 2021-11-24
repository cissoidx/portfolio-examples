#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")
# export POPLAR_ENGINE_OPTIONS='{"autoReport.directory":"./logs/gpt2_report8-2-1","debug.allowOutOfMemory": "true", "autoReport.outputSerializedGraph": "false", "debug.outputAllSymbols": "true", "autoReport.all": "true","profiler.perExecutionStreamCopyCycles":"true"}'
# export POPLAR_ENGINE_OPTIONS='{"profiler.includeFlopEstimates": "true","autoReport.all":"true","debug.allowOutOfMemory": "true", "autoReport.executionProfileProgramRunCount":"1", "autoReport.directory":"./logs/small_training_new"}'
export WANDB_BASE_URL=https://wandb.sourcevertex.net

# Pre-training gpt2 with distributed execution.
# Before running this script:
# 1. Check the hosts ip and partition name ofs server.
# 2. Add host's public key to authorized_keys, make sure that every host can ssh into other hosts.
# 3. Add key of other hosts to known_hosts on each host, this can be done by using 'ssh-keyscan -H <host> >> ./ssh/known_hosts'.
# 4. Synchronize codes, datas, python&sdk environments on all hosts (file system structure must be exactally the same).
# 5. (Optional) Clear the compilation caches on all hosts unless you are sure to use them.
TCP_IF_INCLUDE="enp65s0f0np0"
poprun -vv --host 172.21.16.56,172.21.16.57,172.21.16.58,172.21.16.59,172.21.16.61,172.21.16.62,172.21.16.63,172.21.16.64  \
        --num-instances=8 --num-replicas=32 \
        --ipus-per-replica=4 \
        --vipu-server-host=172.21.16.56 \
        --numa-aware=yes \
        --num-ilds=2 \
        --vipu-partition=p128 \
        --vipu-cluster=c128 \
        --update-partition=no \
        --remove-partition=no \
        --reset-partition=no \
        --print-topology=yes \
        --mpi-global-args="--tag-output --allow-run-as-root  --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
        --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=3600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS" \
        --vipu-server-timeout=3600 \
        --executable-cache-path=/localdata/chaon/tmp \
python train_gpt2.py \
    --model gpt2-medium \
    --layers_per_ipu 0 8 8 8 \
    --gradient_accumulation 512 \
    --epochs 10 \
    --batches_per_step 8 \
    --batch_size 2 \
    --matmul_proportion 0.6 0.6 0.6 0.6 \
    --ipus_per_replica 4 \
    --loss_scaling 50000 \
    --enable_half_partials True \
    --embedding_serialization_factor 1 \
    --recompute_checkpoint_every_layer True \
    --save_model_path './checkpoints/gpt2_medium' 2>&1 | tee logs/gpt2_medium_POD128_$time.log
