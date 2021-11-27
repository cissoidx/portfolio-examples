#!/bin/bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
time=$(date "+%Y%m%d%H%M%S")
# Pre-training gpt2 with distributed execution.
# Before running this script:
# 1. Check the hosts ip and partition name of servers.
# 2. Add host's public key to authorized_keys, make sure that every host can ssh into other hosts.
# 3. Add key of other hosts to known_hosts on each host, this can be done by using 'ssh-keyscan -H <host> >> ./ssh/known_hosts'.
# 4. Synchronize codes, datas, python&sdk environments on all hosts (file system structure must be exactally the same).
# 5. (Optional) Clear the compilation caches on all hosts unless you are sure to use them.
HOST1=`ifconfig eno1 | grep "inet " | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | head -1`
OCT1=`echo "$HOST1" | cut -d "." -f 1`
OCT2=`echo "$HOST1" | cut -d "." -f 2`
OCT3=`echo "$HOST1" | cut -d "." -f 3`
OCT4=`echo "$HOST1" | cut -d "." -f 4`
RNIC1=$OCT1.`expr $OCT2 + 4`.`expr $OCT3`.`expr $OCT4`
RNIC2=$OCT1.`expr $OCT2 + 4`.`expr $OCT3 + 1`.`expr $OCT4`
HOSTS=$RNIC1,$RNIC2
VIPU_SERVER=${VIPU_SERVER:=$RNIC1}
FIRST_PARTITION=`vipu-admin list partitions --api-host $HOST1| grep ACTIVE | cut -d '|' -f 2 | cut -d ' ' -f 2 | head -1`
PARTITON=${PARTITION:=$FIRST_PARTITION}
poprun -vv --host $HOSTS \
        --num-instances=2 --num-replicas=32 \
        --ipus-per-replica=4 \
        --vipu-server-host=$VIPU_SERVER \
        --numa-aware=yes \
        --num-ilds=2 \
        --vipu-partition=$PARTITON \
        --update-partition=yes \
        --remove-partition=no \
        --reset-partition=no \
        --print-topology=yes \
        --mpi-global-args="--tag-output --allow-run-as-root  --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
        --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT=3600 -x POPLAR_LOG_LEVEL=WARN -x POPLAR_SDK_ENABLED -x POPLAR_ENGINE_OPTIONS" \
        --vipu-server-timeout=3600 \
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