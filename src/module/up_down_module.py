import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from functools import partial
from ml_collections.config_dict import ConfigDict

def l2_normalize(x, mask = None):
    _dtype = x.dtype
    x = x.to(torch.float32)
    # x = x + (1.0 - mask).unsqueeze(-1).to(torch.float32) * 1e-6  ##### prevent nan bug
    x = x / torch.maximum(torch.linalg.norm(x, dim=-1, keepdim=True), torch.tensor(1e-6))
    x = x.to(_dtype)
    return x


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


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        global_config,
        dtype,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        self.norm = NormBlock(in_features,global_config)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features,dtype=dtype)
        self.w1 = nn.Linear(in_features, hidden_features,dtype=dtype)
        self.w2 = nn.Linear(hidden_features, in_features,dtype=dtype)
        self._initialize_weights()


    def _initialize_weights(self):
        """Initialize the weights using the specified method."""
        # if self.init_method == "trunc_normal":
        #     init_std = self.init_params[0]  # Standard deviation for initialization
        init_std = 0.02
        nn.init.trunc_normal_(self.w0.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        if self.w0.bias is not None:
            nn.init.zeros_(self.w0.bias)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
            

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x
    



class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dtype=torch.float32):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False,dtype=dtype)
            self.q_bias = nn.Parameter(torch.zeros(in_dim,dtype=dtype))
            self.v_bias = nn.Parameter(torch.zeros(in_dim,dtype=dtype))
            self.register_buffer('zero_k_bias', torch.zeros(in_dim,dtype=dtype))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False,dtype=dtype)
            self.q_bias = nn.Parameter(torch.zeros(out_dim,dtype=dtype))
            self.v_bias = nn.Parameter(torch.zeros(out_dim,dtype=dtype))
            self.register_buffer('zero_k_bias', torch.zeros(out_dim,dtype=dtype))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim, dtype=dtype)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using the specified method."""
        # if self.init_method == "trunc_normal":
        #     init_std = self.init_params[0]  # Standard deviation for initialization
        init_std = 0.02
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        # else: # Default to Xavier uniform
        #     nn.init.xavier_uniform_(self.q_gen.weight)
        #     nn.init.xavier_uniform_(self.k_gen.weight)
        #     nn.init.xavier_uniform_(self.v_gen.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=True)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x
    

class Tokenfactorization(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, global_config, mlp_ratio=2, dtype=torch.float32):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = NormBlock(in_dim,global_config)
        self.attn = CausalAttention(in_dim, out_dim, num_heads, dtype)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = NormBlock(in_dim,global_config)

        self.norm2 = NormBlock(out_dim, global_config)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim,
            global_config= global_config,
            dtype = dtype
        )
    
    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        config,
        dtype
    ):
        super().__init__()
        self.dtype = dtype
        self.projection_out_dim = config.projection_dim
        self.projection_in_dim = config.embedding_dim
        # self.projection = nn.Linear(config.embedding_dim, self.projection_out_dim,dtype=self.dtype)
        # self.blocks = nn.ModuleList()
        self.proj_num_layers = config.proj_num_layers
        dim = self.projection_in_dim
        self.activate_final = config.activate_final
        self.dropout_flag = config.dropout_flag
        layers = []
        for j in range(self.proj_num_layers):
            if j != self.proj_num_layers - 1:
                out_dim = dim
                use_bias = config.use_bias
            else:
                out_dim = self.projection_out_dim
                use_bias = False
            layer = nn.Linear(dim, out_dim, bias=use_bias)
            self.w_init(layer.weight)
            if use_bias:
                self.b_init(layer.bias)
                
            layers.append(layer)
            dim = out_dim
            
        self.blocks = nn.ModuleList(layers)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_rate)
        # self.layer_norm = NormBlock(self.projection_dim)
        
    def w_init(self, weight):
        init_std = 0.02
        nn.init.trunc_normal_(weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        
    def b_init(self, weight):
        torch.nn.init.zeros_(weight)

    def forward(self, x):
        out = x
        num_layers = len(self.blocks)
        for i, layer in enumerate(self.blocks):
            out = layer(out)
            # Apply activation and dropout if condition met
            if i < (num_layers - 1) or self.activate_final:
                if self.dropout_flag and self.dropout is not None:
                    out = self.dropout(out)
                out = self.activation(out)
        out = l2_normalize(out)
        return out