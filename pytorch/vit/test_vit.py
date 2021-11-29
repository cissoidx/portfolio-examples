# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018- The Hugging Face team. All rights reserved.
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


import inspect
import os
import re
import subprocess
import unittest

import pytest
import requests
import torch
import yaml
from attrdict import AttrDict
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

from models import VisionTransformer


def run_vit_cifar10(**kwargs):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    cmd = ["python", "train.py", "--config",
           "b16_cifar10", "--training-steps", "200"]
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.PIPE).decode("utf-8")
    except subprocess.CalledProcessError as e:
        print(f"TEST FAILED")
        print(f"stdout={e.stdout.decode('utf-8',errors='ignore')}")
        print(f"stderr={e.stderr.decode('utf-8',errors='ignore')}")
        raise
    return out


class TestViT(unittest.TestCase):

    @pytest.mark.category3
    def test_final_training_loss(self):
        out = run_vit_cifar10()
        loss = 100.0

        for line in out.split("\n"):
            if line.find("Step: 199/") != -1:
                loss = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-3])
                break
        self.assertGreater(loss, 0.1)
        self.assertLess(loss, 0.91)


class TestViTModel(unittest.TestCase):

    def setUp(self):
        self.url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        self.config = AttrDict({
            'loss': 'CELoss',
            'hidden_size': 768,
            'representation_size': None,
            'attention_probs_dropout_prob': 0.1,
            'dropout_prob': 0.1,
            'drop_path_rate': 0,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'mlp_dim': 3072,
            'num_labels': 1000,
            'patches_size': 16,
            'mixup': False,
        })
        self.inputs = self.prepare_inputs()
        self.model_ref = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224')
        self.model = VisionTransformer(self.config,
                                       img_size=224,
                                       num_labels=1000,
                                       representation_size=None)
        self.model.eval()
        filepath = os.path.join(os.path.dirname(__file__), 'models/keys_match.yaml')
        with open(filepath, mode='r', encoding='UTF-8') as f:
            self.matched_keys = yaml.safe_load(f)

    def prepare_inputs(self):
        image = Image.open(requests.get(self.url, stream=True).raw)
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            'google/vit-base-patch16-224')
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    def hf_inference(self):
        # https://github.com/huggingface/transformers/blob/95f888fd6a30f6d2fc5614347522eb854dcffbd6/tests/test_modeling_vit.py
        outputs = self.model_ref(**self.inputs)
        logits = outputs.logits
        return logits[0][:3]

    def gc_inference(self):
        src = self.model_ref.named_parameters()
        dst = dict(self.model.named_parameters())

        for sname, sparam in src:
            dname = self.matched_keys[sname]
            dst[dname].data.copy_(sparam)

        outputs = self.model(self.inputs['pixel_values'])
        return outputs[0][:3]

    @pytest.mark.category1
    def test_inference(self):
        hf_results = self.hf_inference()
        gc_results = self.gc_inference()
        # [-0.2744, 0.8215, -0.0836]
        self.assertTrue(torch.allclose(hf_results, gc_results, atol=1e-4))
