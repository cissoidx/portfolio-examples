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

import torch
import poptorch
import pytest
import numpy as np
from transformers import BertTokenizer, BertTokenizerFast, GPT2Config, GPT2LMHeadModel
import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


class GTP2Wrapper(torch.nn.Module):
    # Required because poptorch does not support defaults args in the model.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context, past_key_values, labels):
        outputs = self.model.forward(context, past_key_values=past_key_values, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        # loss = poptorch.identity_loss(loss.mean(), reduction="none")
        return loss.mean()


def test_ipu_cpu_match():
    """
    Test that the GPT2 model ran on IPU approximately matches that same
    model ran on the CPU.
    """

    # Config
    batch_size = 1
    config = GPT2Config.from_json_file('config/config.json')
    config.attn_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.summary_first_dropout = 0.0
    config.n_layer = 3
    config.n_embd = 384
    config.n_head = 2

    # Models and options
    opts = poptorch.Options().deviceIterations(1)
    opts.Training.gradientAccumulation(1)
    opts.replicationFactor(1)
    opts.Precision.setPartialsType(torch.float32)
    opts.anchorMode(poptorch.AnchorMode.Final)

    model_cpu = GPT2LMHeadModel(config=config).train()
    model_ipu_ = GPT2LMHeadModel(config=config).train()
    model_ipu_.load_state_dict(model_cpu.state_dict())
    model_ipu = GTP2Wrapper(model_ipu_)

    # Check that copy was successful
    assert model_ipu is not model_cpu
    assert all([(a == b).all() for a, b in zip(
        model_cpu.parameters(), model_ipu.parameters())]) is True

    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=0.001)
    optimizer_ipu = poptorch.optim.AdamW(model_ipu.model.parameters(), lr=0.001, loss_scaling=1.0)
    poptorch_model = poptorch.trainingModel(model_ipu, opts, optimizer=optimizer_ipu)

    # Input
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute yo"
        "Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute Hello, my dog is cute",
        return_tensors="pt")
    labels = torch.ones_like(inputs['input_ids'])
    past_key_values = [(torch.tensor(
        np.random.randn(batch_size, config.n_head, 128 - 1, int(config.n_embd / config.n_head))).type(torch.float),
                        torch.tensor(np.random.randn(batch_size, config.n_head, 128 - 1,
                                                     int(config.n_embd / config.n_head))).type(torch.float))
                       for _ in range(config.n_layer)]

    batch_cpu = (inputs['input_ids'].repeat(batch_size, 1),
                 past_key_values,
                 None,
                 None,
                 None,
                 None,
                 None,
                 None,
                 None,
                 labels.repeat(batch_size, 1))

    batch = (inputs['input_ids'].repeat(1, 1),
             past_key_values,
             labels.repeat(1, 1))

    # Training Loop
    for step in range(10):
        # Step CPU model
        optimizer_cpu.zero_grad()
        for b in range(batch_size):
            cpu_output = model_cpu(*batch_cpu)
            cpu_loss = cpu_output[0]
            cpu_loss.backward()
        optimizer_cpu.step()

        # Step IPU Model
        ipu_output = poptorch_model(*batch)
        ipu_loss = ipu_output

        with torch.no_grad():
            print(f"CPU Loss: {cpu_loss}, IPU Loss: {ipu_loss}")
            # Check the losses are approximately equal
            assert np.allclose(cpu_loss.numpy(), ipu_loss.numpy(), atol=1e-6)


if __name__ == '__main__':
    test_ipu_cpu_match()