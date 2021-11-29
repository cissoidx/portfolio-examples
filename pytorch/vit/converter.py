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
import yaml
import torch

a = torch.load('output/vit-75684.pt')
a = a['model_state_dict']
filepath = os.path.join(os.path.dirname(__file__), 'models/keys_match.yaml')
with open(filepath, mode='r', encoding='UTF-8') as f:
    matched_keys = yaml.safe_load(f)

old_keys = list(a.keys())
print(old_keys)

for k in matched_keys.keys():
    print(k)
    old_k = 'model.' + matched_keys[k]
    a[k] = a[old_k]
    a.pop(old_k, None)

a.pop('classifier.weight', None)
a.pop('classifier.bias', None)


torch.save(a, 'output/vit-in1k.pt')

