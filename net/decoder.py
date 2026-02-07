import torch
from src.module.up_down_module import Tokenfactorization
from net.adaLN import adaLNPreNormTransformerBlock
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config, global_config):
        super(Decoder, self).__init__()
        self.config = config
        self.global_config = global_config
        self.arr_dtype = torch.bfloat16 if self.global_config.bf16_flag else torch.float32
        self.norm_small = self.global_config.norm_small
        self.norm_method = self.global_config.norm_method
        
        self.prefix_up_ = Tokenfactorization(self.config.latent_dim, self.config.dim_feature, self.config.attention_factor_head, global_config=self.global_config, 
                                         mlp_ratio=2, dtype=self.arr_dtype)

        self.transformer_blocks = nn.ModuleList([
            adaLNPreNormTransformerBlock(self.config.causalattention, self.global_config)
            for _ in range(self.config.causalattention.n_layers)
        ])        

    def forward(self, latent_act, tokens_embedding, mask=None, rope_index=None, prefix_positional_embedding=0.):

        batch_size, num_prefix_tokens, dim_latent_feature = latent_act.shape

        B, num_tokens, dim_feature = tokens_embedding.shape

        prefix_act = self.prefix_up_(latent_act)  # (B, N_PRE, F)
        prefix_act = prefix_positional_embedding + prefix_act  # (B, N_PRE, F)
        act = torch.cat([prefix_act, tokens_embedding], dim=1)  # (B, N_PRE + N, ...)
        inf_ = -1e4
        
        suffix_rope_index = torch.arange(num_prefix_tokens, dtype=rope_index.dtype,device=rope_index.device).unsqueeze(0).expand(batch_size, -1) + inf_

        if self.global_config.task != 'evo_reason':
        
            suffix_rope_index = torch.arange(num_prefix_tokens, device=rope_index.device, dtype=rope_index.dtype).unsqueeze(0).expand(batch_size, -1) + inf_

        else:

            suffix_rope_index = torch.full((batch_size, num_prefix_tokens), inf_, dtype=rope_index.dtype, device=rope_index.device)
        
        rope_index = torch.cat([suffix_rope_index, rope_index], dim=1)

        for block in self.transformer_blocks:
            act = block(act, tokens_mask = None, tokens_rope_index =rope_index, cond = None)
        return act