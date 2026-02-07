import numpy as np
from typing import Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.transformer import NormBlock,PowiseFeedForward
from ..module.attention import AttentionEmbedding, AttentionKernel, HyperAttentionEmbedding, PostAttention

class AttentionBlock(nn.Module):
    def __init__(self, config, global_config, hyper_lora_config=None):
        super(AttentionBlock, self).__init__()
        self.config = config
        self.global_config = global_config
        self.hyper_lora_config = hyper_lora_config

        self.attention_embedding = AttentionEmbedding(
            config.attention_embedding, global_config
        )

        if config.hyper_attention_flag:
            self.hyper_attention_embedding = HyperAttentionEmbedding(
                config.hyper_attention_embedding, global_config
            )

        self.attention_kernel = AttentionKernel(
            config.attention_kernel, global_config
        )

        self.post_attention = PostAttention(
            config.post_attention, global_config)

        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, single_act, single_mask=None, rope_index=None, hyper_var=None):
        #### 1. Attention Embedding
        q, k, v = self.attention_embedding(single_act, hyper_var=hyper_var)

        #### 2. HyperAttention Embedding
        if self.config.hyper_attention_flag:
            q, k = self.hyper_attention_embedding(q, k, neighbor_or_rope_idxs = rope_index)
        
        #### 3. Attention Kernel
        out_act = self.attention_kernel(q, k, v, single_mask)

        #### 4. Post Attention
        out_act = self.post_attention(out_act, q)

        #### 5. Dropout
        if self.global_config.dropout_flag:
            out_act = self.dropout(out_act)

        return out_act


class SelfAttention_block(nn.Module):
    def __init__(self, config, global_config):
        super(SelfAttention_block, self).__init__()
        self.config = config
        self.global_config = global_config

        self.attention_block = AttentionBlock(
            config, global_config
        )

        self.norm_block = NormBlock(
            self.config.dim_feature ,global_config
        )

        self.transition_block = PowiseFeedForward(
            self.config.transition, global_config
        )

    def forward(self, q_act, q_rope_index=None,block_mask=None):

        residual, act = q_act, q_act
  
        act = self.norm_block(act)
        
        act = self.attention_block(act, rope_index = q_rope_index, single_mask=block_mask)
        
        act = residual + act

        act = self.transition_block(act)

        return act