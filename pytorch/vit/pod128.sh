# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# VARIABLES TO CHANGE BEFORE RUNNING
# 1. change cluster name and hostnames(or ip address), namely:
# clustername, hostname-server, hostname1, hostname2, etc...
# 2. If you want to use MPI/non-MPI communications, change the following variables
# interface_oob, interface_btl
# 3. make sure write access is allow in /cachedir, or change it according to your settings.

export POPTORCH_CACHE_DIR="/cachedir"
export IPUOF_VIPU_API_TIMEOUT=1000
export POPLAR_LOG_LEVEL=WARN
export POPLAR_ENGINE_OPTIONS='{"target.hostSyncTimeout": "3000"}'
unset IPUOF_CONFIG_PATH

poprun --host [hostname1,hostname2,hostname3,hostname4,hostname5,hostname6,hostname7,hostname8] \
      --vv \
      --numa-aware=yes \
      --update-partition=yes \
      --reset-partition=no \
      --remove-partition=false \
      --vipu-server-host=[hostname-server] \
      --num-ilds=1 --mpi-global-args="--tag-output --allow-run-as-root --output-filename output --mca oob_tcp_if_include [interface_oob] --mca btl_tcp_if_include [interface_btl]" \
      --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x CPATH -x IPUOF_VIPU_API_TIMEOUT -x POPLAR_LOG_LEVEL -x POPLAR_SDK_ENABLED -x POPTORCH_CACHE_DIR -x POPLAR_ENGINE_OPTIONS" \
      --vipu-partition=pod128_vit \
      --vipu-cluster=[cluster_name] \
      --ipus-per-replica 4 \
      --executable-cache-path="/cachedir" \
      --num-replicas=32 --num-instances=8 \
      --vipu-server-timeout=600 \
      python train_from_scratch.py --config b16_in1k_pretrain --prefetch-depth 1 --gradient-accumulation 146 --batch-size 14 --dataloader-workers 96 --rebatch-size 1024

