import torch
import torch.nn as nn
from .encoder import AttentionBlock
from src.module.transformer import NormBlock,PowiseFeedForward

class adaLN(nn.Module):
    def __init__(self, hidden_size, global_config, module, activation='silu'):
        super(adaLN, self).__init__()
        self.hidden_size = hidden_size
        self.global_config = global_config
        self.module = module
        self.activation = activation
        self.arr_dtype = torch.bfloat16 if global_config.bf16_flag else torch.float32
        self.norm = NormBlock(hidden_size,global_config)

    def forward(self, x, cond, other_inputs=()):

        d_act = self.norm(x)

        d_act = self.module(d_act, *other_inputs)

        return d_act
    

class adaLNPreNormTransformerBlock(nn.Module):
    def __init__(self, config, global_config):
        super(adaLNPreNormTransformerBlock, self).__init__()
        self.config = config
        self.global_config = global_config
        self.attention_block = AttentionBlock(self.config.attention, self.global_config)
        self.adaln_block = adaLN(self.config.adaLN.hidden_size, self.global_config, self.attention_block)
        self.transition_block = PowiseFeedForward(self.config.transition, self.global_config)

    def forward(self, tokens, tokens_mask, tokens_rope_index, cond):

        residual,act = tokens,tokens

        act = self.adaln_block(act, cond, other_inputs=(tokens_mask, tokens_rope_index))

        act = act + residual
    
        act = self.transition_block(act)

        return act