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


import numpy as np
import poptorch
import popart
import torch

import popdist


def create_model(opts):
    if opts.use_popdist:
        model_opts = popdist.poptorch.Options()
    else:
        model_opts = poptorch.Options()

    return model_opts


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''

    # Numpy options
    np.random.seed(config.random_seed)

    # Poptorch options
    opts = create_model(config)
    opts.enableExecutableCaching("/localdata/xud/cachedir")

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.batches_per_step)
    if not config.use_popdist:
        opts.replicationFactor(config.replication_factor)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    if config.reduction_type == 'sum':
        opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Sum)
    opts.anchorMode(poptorch.AnchorMode.All)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(True))
    if config.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))
    if config.enable_rts and not config.use_popdist:
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(
                True).minElementsForReplicatedTensorSharding(config.replication_factor))
    opts.randomSeed(config.random_seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    # PopART options
    if config.stochastic_rounding:
        opts.Precision.enableStochasticRounding(True)
    if config.half_partials:
        opts.Precision.setPartialsType(torch.half)

    opts._Popart.set("disableGradAccumulationTensorStreams", True)

    if config.prefetch_depth > 1:
        opts._Popart.set("defaultPrefetchBufferingDepth", config.prefetch_depth)

    if config.recompute_all_stages:
        opts._Popart.set("autoRecomputation", int(popart.RecomputationType.Pipeline))

    # Parallelize optimizer step update across IPUs
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    # opts.Training.setAutomaticLossScaling(True)
    # ALS is experimental
    return opts
