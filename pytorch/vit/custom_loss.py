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


def sigmoid_ce(output, target):
    probs_0 = torch.sigmoid(output)
    probs_1 = 1 - probs_0

    log_probs_0 = - torch.log(probs_0)
    log_probs_1 = - torch.log(probs_1)

    indices_x = torch.arange(0, output.shape[0]).long()
    indices_y = target
    target_scattered_0 = torch.zeros_like(output)
    target_scattered_0.index_put_([indices_x, indices_y], torch.tensor(1.))
    target_scattered_1 = 1 - target_scattered_0

    loss_0 = torch.mul(log_probs_0, target_scattered_0)
    loss_1 = torch.mul(log_probs_1, target_scattered_1)

    loss_0 = torch.sum(loss_0)
    loss_1 = torch.sum(loss_1)

    loss = (loss_0 + loss_1)/output.shape[0]
    return poptorch.identity_loss(loss, reduction="none")


if __name__ == '__main__':
    o = torch.Tensor([[-10, -10, -10, 10],
                      [10, -10, -10, -10]])
    t = torch.Tensor([3, 0]).long()
    loss = sigmoid_ce(o, t)
    print(loss)

    o = torch.Tensor([[-10, ]*1000,
                      [-10, ]*1000])
    t = torch.Tensor([3, 0]).long()
    celoss = torch.nn.CrossEntropyLoss()
    loss = celoss(o, t)
    print(loss)
