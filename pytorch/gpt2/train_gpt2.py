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

import poptorch
import torch
import torch.onnx
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
import sys
import torch
import random
import argparse
import numpy as np
import transformers
from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel
from model.optimized_gpt2_attn import OptimizedGPT2Attention
from model.optimized_gpt2_mlp import OptimizedGPT2MLP
import torch.nn.functional as F
import time
from datetime import datetime
from tqdm import trange, tqdm
import pickle
from utils import load_dataset, _WorkerInit, get_lr_scheduler, get_optimizer, collate_fn, sync_metrics, \
    str_to_bool, outline_attribute, _get_layer_ipu, recomputation_checkpoint, SerializedLinear, calculate_acc, get_generated_datum
from ipu_options import get_options
from poptorch import DataLoader
from poptorch.enums import DataLoaderMode
import logging
import popdist
import popdist.poptorch
import horovod.torch as hvd
import wandb
import pdb

MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}

logging.basicConfig(level=logging.INFO, format="%(message)s")


def logger(msg):
    if not popdist.isPopdistEnvSet() or popdist.getInstanceIndex() == 0:
        logging.info(msg)


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()

    ## model
    parser.add_argument('--model', type=str, default='gpt2', choices=('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'),
                        help='model to train')
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="enable half partials or not")
    parser.add_argument('--save-model-path', default='./checkpoints/model', type=str, required=False,
                        help='model path to save')
    parser.add_argument('--executable-cache-dir', default=None, type=str, required=False,
                        help='executable cache dir')
    parser.add_argument('--training-steps', default=10000, type=int, required=False, help='training steps')
    parser.add_argument('--pretrained-model', default='', type=str, required=False, help='pretrained model path')
    parser.add_argument("--compile-only", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Create an offline IPU target that can only be used for offline compilation.")

    ## dataset
    parser.add_argument('--train-path', default='data/train.pkl', type=str, required=False, help='dataset path')
    parser.add_argument('--max-len', default=128, type=int, required=False, help='max length of input sequence')
    parser.add_argument('--stride', default=128, type=int, required=False, help='stride window size to sample dataset')
    parser.add_argument('--val-num', type=int, default=0, help='validate dataset length')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--num-workers', type=int, default=4, help="workers for dataloader")
    parser.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=False,
                        help="async dataloader")
    parser.add_argument('--vocab-size', type=int, default=30522, help="vocab size if generate mock data")

    ## train
    parser.add_argument('--epochs', default=1, type=int, required=False, help='epochs for training')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default = 1)')
    parser.add_argument('--optimizer', default='AdamW', type=str, required=False, help='optimizer')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False, help='weight_decay')
    parser.add_argument('--learning-rate', default=0.00001, type=float, required=False, help='learning_rate')
    parser.add_argument('--loss-scaling', default=50000.0, type=float, required=False, help='loss_scaling')
    parser.add_argument('--lr-warmup', default=0.1, type=float, required=False, help='lr_warmup')
    parser.add_argument('--lr-schedule', default='constant', type=str, required=False, help='lr_schedule')
    parser.add_argument('--log-steps', default=1, type=int, required=False, help='log_steps')
    parser.add_argument('--gradient-accumulation', default=10, type=int, required=False, help='gradient_accumulation')
    parser.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Set the tensor storage location for optimizer state to be offchip.")
    parser.add_argument("--replicated-tensor-sharding", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable replicated tensor sharding of optimizer state")
    parser.add_argument("--use-wandb", type=str_to_bool, nargs="?", const=True, default=False, help="use wandb or not")

    ## mapping
    parser.add_argument('--layers-per-ipu', type=int, default=3, nargs="+",
                        help='Number of decoder layers per pipeline stage, after the 0th stage (default = 3). Can be a single number, for an equal number decoder layers per IPU.\
                                Or it can be a list of numbers, specifying number of decoder layers for each individual IPU.')
    parser.add_argument('--batches-per-step', default=4, type=int, required=False, help='batches_per_step')
    parser.add_argument('--replication-factor', default=1, type=int, required=False, help='replication_factor')
    parser.add_argument('--ipus-per-replica', default=4, type=int, required=False, help='ipus_per_replica')
    parser.add_argument("--matmul-proportion", type=float, nargs="+",help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--custom-ops", type=str_to_bool, nargs="?", const=True, default=True, help="Enable custom ops")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                             "If True the output of each encoder layer will be stashed keeping the max liveness "
                             "of activations to be at most one layer. "
                             "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                             "The added stash + code could be greater than the reduction in temporary memory.", )
    parser.add_argument("--embedding-serialization-factor", default=1, type=int, help="Matmul serialization factor the embedding layers")
    parser.add_argument("--mlp-serialization-factor", default=1, type=int,
                        help="Matmul serialization factor the mlp layers")

    args = parser.parse_args()
    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        hvd.init()
        args.use_popdist = True
        if popdist.getNumTotalReplicas() != args.replication_factor:
            print(f"The number of replicas is overridden by PopRun. "
                  f"The new value is {popdist.getNumTotalReplicas()}.")
        args.replication_factor = int(popdist.getNumLocalReplicas())
        args.popdist_rank = popdist.getInstanceIndex()
        args.popdist_size = popdist.getNumInstances()

        hvd.broadcast(torch.Tensor([args.seed]), root_rank=0)
    else:
        args.use_popdist = False

    return args


class GTP2Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args.pretrained_model:  # load pretrained model
            sel.model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        else:  # init model
            model_config = MODEL_CONFIG[args.model]
            self.config = GPT2Config.from_json_file(model_config)
            self.model = GPT2LMHeadModel(config=self.config)

        for layer in self.model.transformer.h:
            GPT2Attn = OptimizedGPT2Attention(self.model.config)
            GPT2Attn.load_state_dict(layer.attn.state_dict())
            layer.attn = GPT2Attn

            if args.mlp_serialization_factor > 1:
                GPT2MLP = OptimizedGPT2MLP(self.model.config, args.mlp_serialization_factor)
                layer.mlp = GPT2MLP


        if args.embedding_serialization_factor > 1:
            serialized_lmhead = SerializedLinear(self.config.n_embd, self.config.vocab_size,
                                                 args.embedding_serialization_factor,
                                                 bias=False,
                                                 mode=poptorch.MatMulSerializationMode.OutputChannels)
            serialized_lmhead.load_state_dict(self.model.lm_head.state_dict())
            self.model.lm_head = serialized_lmhead
            self.model.tie_weights()

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.model.transformer.wte = poptorch.BeginBlock(self.model.transformer.wte, "wte", ipu_id=0)
        self.model.transformer.wpe = poptorch.BeginBlock(self.model.transformer.wpe, "wpe", ipu_id=0)
        outline_attribute(self.model.transformer.ln_f, "LayerNorm")

        layer_ipu = _get_layer_ipu(args.layers_per_ipu)
        for index, layer in enumerate(self.model.transformer.h):
            ipu = layer_ipu[index]
            if args.recompute_checkpoint_every_layer:
                recomputation_checkpoint(layer)
            self.model.transformer.h[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Layer {index:<2} --> IPU {ipu}")

        logger(f'LM_head --> IPU 0')
        self.model.lm_head = poptorch.BeginBlock(self.model.lm_head, ipu_id=0)

    def forward(self, input_ids, labels):
        transformer_outputs = self.model.transformer(input_ids=input_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        loss = poptorch.identity_loss(loss, reduction="none")

        acc = calculate_acc(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss, acc


if __name__ == "__main__":
    args = set_args()
    opts = get_options(args)

    # W&B
    if args.use_wandb and (not args.use_popdist or args.popdist_rank == 0):
        wandb.init(project="torch-gpt2", settings=wandb.Settings(console="wrap"),
                   name='model-' + str(args.model) + ' ipus-' + \
                        str(args.ipus_per_replica) + ' bs-' + str(args.batch_size))
        wandb_config = vars(args)
        wandb.config.update(wandb_config)

    # Dataloader
    logger("------------------- Data Loading Started ------------------")
    start_loading = time.perf_counter()
    train_dataset, validate_dataset = load_dataset(logger, args)
    loader = DataLoader(opts,
                        train_dataset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        worker_init_fn=_WorkerInit(args.seed),
                        collate_fn=collate_fn,
                        drop_last=True,
                        auto_distributed_partitioning=not isinstance(train_dataset, torch.utils.data.IterableDataset),
                        mode=DataLoaderMode.AsyncRebatched if args.async_dataloader else DataLoaderMode.Sync)
    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    logger(f"Data loaded in {duration_loader} secs")
    logger("-----------------------------------------------------------")

    model = GTP2Wrapper(args).half().train()

    optimizer = get_optimizer(args.optimizer, args.weight_decay, args.learning_rate, args.loss_scaling, model,
                              use_popdist=args.use_popdist)
    scheduler = get_lr_scheduler(optimizer, args.lr_schedule, args.lr_warmup, steps_per_epoch * args.epochs)

    poptorch_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    logger("---------- Compilation/Loading from Cache Started ---------")
    start_compile = time.perf_counter()
    datum = get_generated_datum(args)
    poptorch_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    logger(f"Compiled/Loaded model in {duration_compilation} secs")
    logger("-----------------------------------------------------------")

    # Save model and end here if compile only mode is enabled
    if args.compile_only:
        logger("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    # Training loop
    steps_finished = 0
    logger("--------------------- Training Started --------------------")
    factor = args.gradient_accumulation * args.batches_per_step
    start_train = time.perf_counter()

    total_step = 0
    for epoch in range(args.epochs):
        for batch_idx, (input_ids, labels) in enumerate(loader):
            _labels = labels[:, 1:]
            labels = torch.cat((_labels, -100 * torch.ones((_labels.size(0), 1), dtype=torch.long)), dim=1)
            start_step = time.perf_counter()
            outputs = poptorch_model(input_ids=input_ids, labels=labels)

            scheduler.step()
            poptorch_model.setOptimizer(optimizer)
            step_length = sync_metrics(time.perf_counter() - start_step)
            outputs_sync = sync_metrics(outputs, factor)
            num_instances = args.popdist_size if args.use_popdist else 1
            step_throughput = num_instances * args.replication_factor * args.batch_size * args.gradient_accumulation * \
                              args.batches_per_step / step_length
            if (batch_idx + 1) % args.log_steps == 0:
                logger("batch {} of epoch {}, loss: {}, acc: {}, lr: {}, Throughput: {} seq/s".format(
                    batch_idx + 1, epoch + 1, outputs_sync[0], outputs_sync[1], scheduler.get_last_lr()[0],
                    step_throughput))

            total_step += 1
            if args.use_wandb and (not args.use_popdist or args.popdist_rank == 0):
                wandb.log({"Loss": outputs_sync[0],
                           "Acc": outputs_sync[1],
                           "LR": scheduler.get_last_lr()[0],
                           "Step": total_step,
                           "Epoch": epoch,
                           "Throughput": step_throughput})

    if args.save_model_path:
        if not args.use_popdist or args.popdist_rank == 0:
            model_path = os.path.join(args.save_model_path, 'model'.format(epoch + 1))
            logger('saving current model to {}'.format(model_path))
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            model_to_save = model.model.module if hasattr(model, 'module') else model.model
            model_to_save.save_pretrained(model_path)
