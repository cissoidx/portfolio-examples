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

import os
from pathlib import Path
from functools import partial

import poptorch
import torch
import torchvision
from torch.utils.data import Dataset, IterableDataset
from dataset.preprocess import get_preprocessing_pipeline
from dataset.mixup_utils import mixup_data


def get_random_datum(config):
    result = []
    dataset = GeneratedDataset(shape=[3, 224, 224], size=config.samples_per_step,
                               half_precision=True)
    data = (dataset[i] for i in range(config.samples_per_step))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


class GeneratedDataset(Dataset):
    """
    Generated dataset creates a random dataset with the given shape and precision.
    The size determines the number of items in the dataset.
    """

    def __init__(self, shape, size=60000, half_precision=False):
        self.size = size
        self.half_precision = half_precision
        self.data_shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        synthetic_data = torch.rand(self.data_shape)
        synthetic_label = torch.randint(0, 2, [1], dtype=torch.long)
        if self.half_precision:
            synthetic_data = synthetic_data.half()
        return synthetic_data, synthetic_label


def get_data(configs, model_opts, train=True, async_dataloader=False):
    """
    A factory method to create a dataloader responsible for sending data
    to the IPU device. This build the appropriate dataset and wraps it in a dataloader.
    """
    if configs.precision[:3] == "16.":
        half_precision = True
    elif configs.precision[:3] == "32.":
        half_precision = False
    transform = get_preprocessing_pipeline(
        train, 224, half_precision, normalize=True, extra_aug=configs.extra_aug)
    # Determine the size of the small datasets
    if hasattr(configs, "iterations"):
        dataset_size = configs.batch_size * \
            model_opts.device_iterations * \
            configs.replication_factor * \
            model_opts.Training.gradient_accumulation * \
            configs.iterations

    # Select the right dataset
    if configs.dataset in ["synthetic", "generated"]:
        if hasattr(configs, "iterations"):
            dataset = GeneratedDataset(
                224, size=dataset_size, half_precision=half_precision)
        else:
            dataset = GeneratedDataset(224, half_precision=half_precision)
    elif configs.dataset in ["imagenet1k", "imagenet21k"]:
        data_folder = 'train' if train else 'validation'
        dataset = torchvision.datasets.ImageFolder(os.path.join(
            configs.input_files, data_folder), transform=transform)
    elif configs.dataset == "cifar10":
        data_path = Path(__file__).parent.parent.absolute().joinpath(
            "data").joinpath("cifar10")
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train,
                                               download=True, transform=transform)
    else:
        raise Exception('Dataset type not recognized: %s' % configs.dataset)

    if async_dataloader:
        if configs.global_batch_size == 1:
            # avoid rebatch overhead
            mode = poptorch.DataLoaderMode.Async
        else:
            mode = poptorch.DataLoaderMode.AsyncRebatched
    else:
        mode = poptorch.DataLoaderMode.Sync
    
    dataloader = poptorch.DataLoader(model_opts,
                                     dataset,
                                     batch_size=configs.batch_size if not(isinstance(
                                         dataset, IterableDataset)) else None,
                                     num_workers=configs.dataloader_workers,
                                     shuffle=train and not(isinstance(dataset, IterableDataset)),
                                     drop_last=not(isinstance(dataset, IterableDataset)),
                                     persistent_workers=True,
                                     auto_distributed_partitioning=not isinstance(
                                         dataset, IterableDataset),
                                     worker_init_fn=None,
                                     mode=mode,
                                     async_options={'load_indefinitely': True},
                                     rebatched_worker_size=configs.rebatch_size if async_dataloader else None,
                                     collate_fn=partial(collate_fn, alpha=configs.alpha) if train and configs.mixup else None)
    return dataloader


def collate_fn(batch, alpha):
    inputs = []
    targets = []
    for x, y in batch:
        inputs += [x]
        targets += [torch.tensor(y)]
    input_data = torch.stack(inputs)
    labels = torch.stack(targets)
    # print(input_data.shape, labels.shape)
    mixed_x, y_a, y_b, lam = mixup_data(input_data, labels, alpha)
    # print(mixed_x.shape, y_a.shape, y_b.shape, lam.shape)
    return mixed_x, y_a, y_b, lam


if __name__ == '__main__':
    opts = poptorch.Options()
    transform = get_preprocessing_pipeline(True, 224, True, True, False)
    dataset = torchvision.datasets.ImageFolder('/localdata/ai-datasets/imagenet1k/validation',
                                               transform=transform)
    # data = next(iter(dataset))
    # print(data[0].shape, data[1])
    dataloader = poptorch.DataLoader(opts,
                                     dataset,
                                     batch_size=8,
                                     num_workers=2,
                                     shuffle=True,
                                     drop_last=True,
                                     persistent_workers=True,
                                     auto_distributed_partitioning=False,
                                     worker_init_fn=None,
                                     mode=poptorch.DataLoaderMode.Async,
                                     async_options={'load_indefinitely': True},
                                     collate_fn=partial(collate_fn, alpha=0.5))
    data = next(iter(dataloader))
    print(len(data))  # mixed_x, y_a, y_b, lam
