import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module.transformer import NormBlock
from .selfattention import SelfAttention_block
from torch.nn.attention.flex_attention import create_block_mask



class SuffixEncoder(nn.Module):
    def __init__(self, config, global_config):
        super(SuffixEncoder, self).__init__()
        self.config = config
        self.global_config = global_config
        self.norm_small = self.global_config.norm_small
        self.norm_method = self.global_config.norm_method
        self.arr_dtype = torch.bfloat16 if self.global_config.bf16_flag else torch.float32
        self.prefix_embedding_table = nn.Embedding(
            num_embeddings=self.config.num_prefix_tokens,
            embedding_dim=self.config.dim_feature,
            dtype=torch.float32
        )
### add prefix_positional_embedding
        self.prefix_position_embedding_table = nn.Embedding(
            num_embeddings=self.config.num_prefix_tokens,
            embedding_dim=self.config.dim_feature,
            dtype=torch.float32
        )
        self.transformer_blocks = nn.ModuleList([
            SelfAttention_block(self.config.selfattention, self.global_config)
            for _ in range(self.config.selfattention.n_layers)
        ])

        self.head_num = self.config.selfattention.attention_embedding.n_head    

        self.suffix_causal_flag = self.global_config.suffix_causal_flag

        # self.norm_block = NormBlock(self.config.dim_feature,self.global_config)

    def forward(self, tokens_embedding, tokens_mask, tokens_rope_index):
        ### Inputs: tokens: (B, T, F), cond: (B, F)
        ### Returns: act: (B, T, F)
        device = tokens_embedding.device

        batch_size, num_tokens, dim_feature = tokens_embedding.shape

        num_prefix_tokens = self.config.num_prefix_tokens
        prefix_tokens = torch.arange(num_prefix_tokens, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(device)
        prefix_embedding = self.prefix_embedding_table(prefix_tokens).to(tokens_embedding.dtype)
        prefix_position_embedding = self.prefix_position_embedding_table(prefix_tokens).to(tokens_embedding.dtype)

        inf_ = -1e4

        if self.global_config.task != 'evo_reason':
        
            suffix_rope_index = torch.arange(num_prefix_tokens, device=device, dtype=tokens_rope_index.dtype).unsqueeze(0).expand(batch_size, -1) + inf_

        else:
            suffix_rope_index = torch.full((batch_size, num_prefix_tokens), inf_, dtype=tokens_rope_index.dtype, device=device)
        
        rope_index = torch.cat([tokens_rope_index, suffix_rope_index], dim=1)
        
        act = torch.cat([tokens_embedding, prefix_embedding+prefix_position_embedding], dim=1)

        suffix_mask = torch.full((batch_size, num_prefix_tokens), 1, dtype=tokens_rope_index.dtype, device=device)

        full_mask = torch.cat([tokens_mask, suffix_mask], dim=1)

        full_mask_for_attention = full_mask.unsqueeze(1).expand(-1, self.head_num, -1)
        full_mask_for_attention = full_mask_for_attention.unsqueeze(2).expand(-1, -1, full_mask.size(1), -1)
        attention_mask = full_mask_for_attention & full_mask_for_attention.transpose(2, 3)
        
        if self.suffix_causal_flag:
            suffix_causal_mask = torch.tril(torch.ones((num_prefix_tokens,num_prefix_tokens), device=attention_mask.device))
            attention_mask[:,:, -num_prefix_tokens:, -num_prefix_tokens:] = suffix_causal_mask

        attention_mask = attention_mask.to(torch.bool)

        for block in self.transformer_blocks:
            act = block(act, rope_index,block_mask = attention_mask)

        return act[:, -num_prefix_tokens:,:],prefix_position_embedding