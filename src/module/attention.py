import numpy as np
from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import math

_MAX_WAVELENGTH = 10000

def segment_mask(
    q_segment_ids: torch.Tensor,
    kv_segment_ids: torch.Tensor,
):
    # [B, T, 1] or [T, 1]
    q_segment_ids = q_segment_ids.unsqueeze(-1)
    # [B, 1, S] or [1, S]
    if kv_segment_ids.ndim == 1:
        kv_segment_ids = kv_segment_ids.unsqueeze(0)
    else:
        kv_segment_ids = kv_segment_ids.unsqueeze(1)
    return (q_segment_ids == kv_segment_ids).bool()

def _attention(
    q: torch.Tensor, # (B, N, H, C)
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: Optional[torch.Tensor],
    sm_scale: float = 1.0,
    causal: bool = False,
    attention_type: str = 'self',
):
    n_seq_q = q.shape[-3]
    n_seq_k = k.shape[-3]
    logits = torch.einsum('bqhc,bkhc->bhqk', q, k)

    # sm_scale = torch.tensor(sm_scale, dtype=logits.dtype, device=logits.device)

    mask = None
    if segment_ids is not None:
        if attention_type == 'self':
            mask = segment_mask(segment_ids, segment_ids).unsqueeze(1)
        elif attention_type == 'cross':
            mask = segment_mask(*segment_ids).unsqueeze(1)
        mask = mask.expand(logits.shape)
    if causal:
        causal_mask = torch.tril(torch.ones((1, 1, n_seq_q, n_seq_k), dtype=torch.bool, device=logits.device))
        causal_mask = causal_mask.expand(logits.shape)
        mask = causal_mask if mask is None else mask & causal_mask

    logits = logits if mask is None else torch.where(mask, logits, float("-inf"))
    weights = F.softmax(logits * sm_scale, dim=-1).type_as(q)
    return torch.einsum('bhqk,bkhc->bqhc', weights, v)


def apply_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    max_wavelength: int = 10000,
):
    """apply rope for input.
    Inputs:
        inputs: shape of (B, N, H, C), input;
        positions: shape of (B, N), position index;
        head_dim: head dimension;
        max_wavelength: max wavelength;
    Returns:
        out: shape of (B, N, H, C), output;
    """
    # head_dim 是 n_channel 也就是 C
    fraction = (2 * torch.arange(0, head_dim // 2) / head_dim).float()
    timescale = max_wavelength ** fraction
    timescale = timescale.to(inputs.device)
    sinusoid_inp = (positions[..., None] / timescale[None, None, :]).float()
    sinusoid_inp = sinusoid_inp[..., None, :]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    first_half, second_half = torch.split(inputs, inputs.size(-1) // 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = torch.cat([first_part, second_part], dim=-1)

    out = out.to(inputs.dtype)

    return out


class AttentionEmbedding(nn.Module):

    def __init__(self, config, global_config):
        super(AttentionEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config
        self.arr_dtype = torch.bfloat16 if self.global_config.bf16_flag else torch.float32

        self.attention_type = self.config.attention_type
        assert self.attention_type in ['self', 'cross'], "attention_type must be either 'self' or 'cross'."
        self.dim_feature = int(self.config.dim_feature)
        self.n_head = int(self.config.n_head)

        assert self.dim_feature % self.n_head == 0, "dim_feature must be divisible by n_head."
        self.n_channel = self.dim_feature // self.n_head
        self.embedding_pair_flag = self.config.embedding_pair_flag

        DenseModule = nn.Linear
        arg_dict = {'in_features': self.dim_feature, 'out_features': self.dim_feature, 'bias': False,"dtype": self.arr_dtype}

        self.q_gen = DenseModule(**arg_dict)
        self.k_gen = DenseModule(**arg_dict)
        self.v_gen = DenseModule(**arg_dict)
        if self.embedding_pair_flag:
            self.z_gen = DenseModule(**arg_dict)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights using the specified method."""
        # if self.init_method == "trunc_normal":
        #     init_std = self.init_params[0]  # Standard deviation for initialization
        init_std = 0.02
        nn.init.trunc_normal_(self.q_gen.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.k_gen.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        nn.init.trunc_normal_(self.v_gen.weight, mean=0.0, std=init_std, a=-2.*init_std, b=2.*init_std)
        # else: # Default to Xavier uniform
        # nn.init.xavier_uniform_(self.q_gen.weight)
        # nn.init.xavier_uniform_(self.k_gen.weight)
        # nn.init.xavier_uniform_(self.v_gen.weight)


    def forward(self, single_act, pair_act=None, hyper_var=None):
        if self.attention_type == "self":
            s_i = single_act
            batch_size, n_q, _ = s_i.shape
            n_k = n_q
            q = self.q_gen(s_i) # (B, Q, H*C)
            k = self.k_gen(s_i)
            v = self.v_gen(s_i)
        else: # cross
            sq_i, sk_i = single_act
            batch_size, n_q, _ = sq_i.shape
            _, n_k, _ = sk_i.shape
            q = self.q_gen(sq_i) # (B, Q, H*C)
            k = self.k_gen(sk_i) # (B, K, H*C)
            v = self.v_gen(sk_i)

        q = q.view(batch_size, n_q, self.n_head, self.n_channel)
        k = k.view(batch_size, n_k, self.n_head, self.n_channel)
        v = v.view(batch_size, n_k, self.n_head, self.n_channel)

        return q, k, v
    

class HyperAttentionEmbedding(nn.Module):

    def __init__(self, config, global_config):
        super(HyperAttentionEmbedding, self).__init__()
        self.config = config
        self.global_config = global_config

    def forward(self, q_i, k_i, m_j=None, z_ij=None, m_ij=None, neighbor_or_rope_idxs=None):
        """
        Modifies q, k and v with hyper-attention kernel.
        Inputs:
            q_i: shape of (B, N, H, C), query;
            k_i: shape of (B, N, H, C), key;
            m_j: shape of (B, N, n) or (B, N, N), mask for neighbor;
            m_ij: shape of (B, N, N) or (B, N, n), mask for pair activation;
            z_ij: shape of (B, N, N, H, C) or (B, N, n, H, C), pair activation;
            neighbor_or_rope_idxs:
                for HAK embedding, neighbor_idxs:
                    shape of (B, N, n), neighbor index for sparse mode, only used for sparse mode;
                for RoPE embedding, rope_indexs: 
                    shape of (B, N), position index for RoPE;
        Outputs:
            q_hyp: shape of (B, N, H, C), modified query;
            k_hyp: shape of (B, N, H, C), modified key;
        NOTE: 
            This module is only used for self-attention, and so Q/K = N;
        """

        # basic config
        arr_type = torch.bfloat16 if self.global_config.bf16_flag else torch.float32

        kernel_type = self.config.kernel_type
        assert kernel_type in ['hak', 'rope'], "kernel_type must be either 'hak' or 'rope'."

        batch_size, n_seq, n_head, n_channel = q_i.shape
        assert k_i.shape == q_i.shape, "q and k must have the same shape."
        
        # RoPE Kernel
        if kernel_type == 'rope':
            if self.config.split_rope_flag:
                q_i, q_i_rope = torch.split(q_i, q_i.size(-2)//2, dim=-2)
                k_i, k_i_rope = torch.split(k_i, k_i.size(-2)//2, dim=-2)
                q_i_rope = apply_rope(q_i_rope, neighbor_or_rope_idxs, q_i.size(-1))
                k_i_rope = apply_rope(k_i_rope, neighbor_or_rope_idxs, q_i.size(-1))
                q_hyp = torch.cat([q_i, q_i_rope], dim=-2)
                k_hyp = torch.cat([k_i, k_i_rope], dim=-2)
            else:
                q_hyp = apply_rope(q_i, neighbor_or_rope_idxs, q_i.size(-1))
                k_hyp = apply_rope(k_i, neighbor_or_rope_idxs, q_i.size(-1))

        
        return q_hyp, k_hyp
    

class AttentionKernel(nn.Module):

    def __init__(self, config, global_config):
        super(AttentionKernel, self).__init__()
        self.config = config
        self.global_config = global_config
        # self.dropout_rate = self.config.dropout_rate

    def forward(self, q, k, v, m=None):
        """Attention operation with optional bias and mask.
        Inputs:
            q: shape of (B, Q, H, C), query;
            k: shape of (B, K, H, C), key; 
            v: shape of (B, K, H, C), value;
            b: shape of (B, H, Q, K), bias;
            m: Flex attention block mask;

            out: shape of (B, Q, H, C);
        """

        # basic config
        causal_flag = self.config.causal_flag
        flash_attention_flag = self.config.flash_attention_flag

        sm_scale = 1. / math.sqrt(q.shape[-1])

        if m is not None:
            if flash_attention_flag:
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()
                out = F.scaled_dot_product_attention(q, k, v,attn_mask=m)
                out = out.permute(0, 2, 1, 3).contiguous()
                
            else:
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()
                out = F.scaled_dot_product_attention(q, k, v,attn_mask=m)
                out = out.permute(0, 2, 1, 3).contiguous()
        else:
            if flash_attention_flag:
                raise NotImplementedError("flash attention without mask is not implemented yet.")
            else:
                
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()
                out = F.scaled_dot_product_attention(q, k, v,is_causal=causal_flag)
                out = out.permute(0, 2, 1, 3).contiguous()
        return out
    

class PostAttention(nn.Module):
    def __init__(self, config, global_config):
        super(PostAttention, self).__init__()
        self.config = config
        self.global_config = global_config
        self.out_dim = config.out_dim
        self.gating_flag = config.gating_flag
        self.arr_dtype = torch.bfloat16 if global_config.bf16_flag else torch.float32
        self.dropout_flag = global_config.dropout_flag

        if self.gating_flag:
        ## config.n_channel * config.n_head, = config.out_dim
            self.gating = nn.Linear(config.n_channel * config.n_head, config.n_channel * config.n_head, bias=True, dtype=self.arr_dtype)
            nn.init.zeros_(self.gating.weight)
            nn.init.ones_(self.gating.bias)

        self.output = nn.Linear(config.n_channel * config.n_head, self.out_dim, bias=True, dtype=self.arr_dtype)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x, q=None):
        batch_size, n_seq, n_head, n_channel = x.shape
        x = x.reshape(batch_size, n_seq, -1).to(self.arr_dtype)
        if self.gating_flag:
            gating_values = self.gating(q.reshape(batch_size, n_seq, -1))
            gating_values = torch.sigmoid(gating_values.float()).to(self.arr_dtype)
            x = x * gating_values

        x = self.output(x)
        return x
    

