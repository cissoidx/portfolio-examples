import torch
import torch.nn as nn
import poptorch
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from utils import SerializedLinear

class OptimizedGPT2MLP(GPT2MLP):
    def __init__(self, config, serialization_factor):
        super().__init__(config.n_inner, config)
        self.serialized_c_fc = SerializedLinear(config.hidden_size, config.n_inner,
                                             serialization_factor,
                                             bias=True,
                                             mode=poptorch.MatMulSerializationMode.OutputChannels)
        self.serialized_c_fc.weight = nn.Parameter(self.c_fc.weight.T)
        self.serialized_c_fc.bias = nn.Parameter(self.c_fc.bias)
        # self.serialized_c_fc.load_state_dict(self.c_fc.state_dict())

        self.serialized_c_proj = SerializedLinear(config.n_inner, config.hidden_size,
                                             serialization_factor,
                                             bias=True,
                                             mode=poptorch.MatMulSerializationMode.OutputChannels)
        self.serialized_c_proj.weight = nn.Parameter(self.c_proj.weight.T)
        self.serialized_c_proj.bias = nn.Parameter(self.c_proj.bias)
        # self.serialized_c_proj.load_state_dict(self.c_proj.state_dict())

    def forward(self, hidden_states):
        hidden_states = self.serialized_c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.serialized_c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states