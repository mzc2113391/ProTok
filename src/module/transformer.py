import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections.config_dict import ConfigDict

class LayerNormBlock(nn.Module):
    def __init__(self,out_dim,global_config,norm_method="layernorm"):
        super(LayerNormBlock, self).__init__()
        self.norm_method = norm_method
        self.eps = global_config.norm_small
        self.out_dim = out_dim

        if self.norm_method == "layernorm":
            self.norm = nn.LayerNorm(normalized_shape=self.out_dim, eps=self.eps)
        elif self.norm_method == "rmsnorm":
            self.norm = nn.RMSNorm(normalized_shape=self.out_dim,eps=self.eps)
        else:
            raise ValueError(f"Unsupported norm method: {self.norm_method}")

    def forward(self, x):
        x_safe = x.to(torch.float32)
        x_safe = self.norm(x_safe)
        x_safe = x_safe.to(x.dtype)
        return x_safe

## use Gemma2 RMSnorm https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class NormBlock(nn.Module):

    def __init__(
        self,
        out_dim: int,
        global_config: ConfigDict,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = global_config.norm_small
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(out_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)
    

    
### use feeddorward from GemmaDecoderlayer: https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class PowiseFeedForward(nn.Module):
    def __init__(self, config,global_config):
### activation function: gelu 
        super(PowiseFeedForward, self).__init__()
        self.config = config
        self.global_config = global_config
        dim_channel = self.config.dim_channel
        d_ff = self.config.transition_factor * dim_channel
        self.arr_dtype = torch.bfloat16 if global_config.bf16_flag else torch.float32
        self.mlp = GemmaMLP(hidden_size=dim_channel, intermediate_size=d_ff, dtype=self.arr_dtype) 
        self.normblock = NormBlock(dim_channel,global_config)
        self.dropout = nn.Dropout(self.config.dropout_rate)
    def forward(self, x):
        act, d_act = x, x
        d_act = self.normblock(d_act)
        d_act = self.mlp(d_act)
        if self.global_config.dropout_flag:
            d_act = self.dropout(d_act)
        act = act + d_act       
        return act

## use GemmaMLP from https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype
    ):
        super().__init__()
        self.dtype = dtype
        self.gate_proj = nn.Linear(hidden_size, intermediate_size,dtype=self.dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size,dtype=self.dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size,dtype=self.dtype)
        self._initialize_weights() 

    def _initialize_weights(self):
        """Initialize the weights of linear layers.
        
        Supports the following initialization methods:
        - "trunc_normal": Truncated normal initialization with configurable standard deviation
        - Default: Xavier uniform initialization
        
        Biases are initialized to zeros if they exist.
        """
        init_std = 0.02
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs