import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.transformer import NormBlock
from net.decoder import Decoder
from src.model.prefixtransformer import SuffixEncoder as Encoder
import numpy as np
from src.module.up_down_module import Tokenfactorization,ProjectionHead
from src.common.utils import DecoderUtils,l2_normalize,_format_beam_results, _format_sample_results,default_embed_init
import lightning as L
from typing import Any, Optional, Sequence, Union
    

class ProTok(nn.Module):
    def __init__(self, config, global_config, token_codebook):
        super(ProTok, self).__init__()
        self.config = config
        self.global_config = global_config
        self.token_codebook = token_codebook
        self.arr_dtype = torch.bfloat16 if self.global_config.bf16_flag else torch.float32
        self.common_embedding_table = nn.Parameter(
            torch.empty(4, self.config.dim_feature, dtype=torch.float32)
        )
        default_embed_init(self.common_embedding_table)
        self.protoken_embedding_dense = nn.Linear(
            self.token_codebook.shape[1], self.config.dim_feature, bias=False,dtype=torch.float32
        )
        self.encoder = Encoder(self.config.encoder, self.global_config)
        self.num_prefix_tokens = self.config.encoder.num_prefix_tokens
        self.encoder_down = Tokenfactorization(
            self.config.dim_feature, self.config.vq_config.latent_dim, self.config.attention_factor_head, self.global_config, mlp_ratio=2, dtype=self.arr_dtype
        )
        nn.init.xavier_uniform_(self.protoken_embedding_dense.weight)
        if self.global_config.task != 'evo_reason':
            self.latent_norm_block = NormBlock(self.config.vq_config.latent_dim, self.global_config)
        self.decoder = Decoder(self.config.decoder, self.global_config)
        self.clip_projection_layer = ProjectionHead(self.config.clip, dtype = self.arr_dtype)

    def forward(self, inputs):
        ######### embedding #########
        self.token_codebook = self.token_codebook.to(self.protoken_embedding_dense.weight.device)
        protoken_embedding_table = self.protoken_embedding_dense(self.token_codebook)
        protoken_input_embedding_table = torch.cat(
            [protoken_embedding_table, self.common_embedding_table], dim=0
        )
        protoken_embedding = F.embedding(
            inputs['encoder_protokens'], protoken_input_embedding_table
        )
        tokens_embedding = protoken_embedding 
        tokens_embedding = tokens_embedding.to(self.arr_dtype)
        ######### encoding #########
        encode_act, prefix_positional_embedding = self.encoder(
            tokens_embedding, inputs['encoder_mask'], inputs['encoder_residue_index']
        )
        latent_act = self.encoder_down(encode_act)
        B,Q,D = latent_act.shape
        if self.global_config.task != 'evo_reason':
            latent_act = self.latent_norm_block(latent_act)
        latent_act_to_clip = latent_act.reshape(B,-1)
        latent_act_to_clip = l2_normalize(latent_act_to_clip)
        latent_clip = self.clip_projection_layer(latent_act_to_clip)
        latent_clip = latent_clip.to(torch.float32)
        quantized = latent_act
        decoder_protokens_embedding = F.embedding(
            inputs['decoder_protokens'], protoken_input_embedding_table
        )
        decoder_protokens_embedding = decoder_protokens_embedding.to(self.arr_dtype)
        decode_act = self.decoder(
            quantized, decoder_protokens_embedding, mask = None, rope_index=inputs['decoder_residue_index'], prefix_positional_embedding=prefix_positional_embedding
        )
        decoder_embedding_table = torch.cat([protoken_embedding_table, self.common_embedding_table[0:3]], dim=0)
        decode_act = decode_act.to(torch.float32)
        decode_protokens_logits = torch.einsum('bnd,vd->bnv', decode_act, decoder_embedding_table)
        outputs = {
            'decode_protokens_logits': decode_protokens_logits,
            "quantized": latent_act,
            "latent_clip" : latent_clip,
            "prefix_positional_embedding":prefix_positional_embedding,
            "decoder_act": decode_act,
        }
        return outputs
        

    @torch.no_grad()
    def decode_batch(
        self,
        latent_act: torch.Tensor,
        method: str = "beam_search",
        max_len: Optional[int] = None,
        min_len: int = 64,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        forbidden_token_ids: Optional[Union[Sequence[int], torch.Tensor]] = (20, 21, 22, 24),
        num_prefix: Optional[int] = None,
        eos_warmup: int = 0,
        eos_bias: float = 0.0,
    ):

        device, dtype = latent_act.device, self.arr_dtype
        B = latent_act.shape[0]
        num_prefix = self.num_prefix_tokens if num_prefix is None else num_prefix
        if max_len is None:
            max_len = 1024
        if latent_act.ndim != 3 or latent_act.shape[1] != num_prefix or num_prefix != self.num_prefix_tokens:
            raise ValueError("Latent prefix dimension must match the checkpoint.")
        if method not in ("beam_search", "greedy_search", "sample", "top_p", "top_k"):
            raise ValueError(f"Unknown decoding method: {method}")
        if num_beams < 1 or num_return_sequences < 1:
            raise ValueError("num_beams and num_return_sequences must be positive.")
        if not 0 <= min_len <= max_len or max_len < 1:
            raise ValueError("Require 0 <= min_len <= max_len and max_len > 0.")
        bos_id, eos_id = 22, 23
        

        self.token_codebook = self.token_codebook.to(device)
        protoken_emb = self.protoken_embedding_dense(self.token_codebook)
        input_table = torch.cat([protoken_emb, self.common_embedding_table], dim=0)
        output_table = torch.cat([protoken_emb, self.common_embedding_table[:3]], dim=0)
        vocab_size = output_table.shape[0]


        is_beam = (method == "beam_search")
        is_sample = method in ("sample", "top_p", "top_k")
        group_size = num_beams if is_beam else (int(num_return_sequences) if is_sample else 1)
        batch_total = B * group_size

        latent_exp = latent_act.repeat_interleave(group_size, dim=0)
        prefix_tokens = torch.arange(num_prefix, device=device).unsqueeze(0).expand(batch_total, -1)
        prefix_pos_emb = self.encoder.prefix_position_embedding_table(prefix_tokens).to(dtype)
        
        inputs = torch.full((batch_total, 1), bos_id, dtype=torch.long, device=device)
        is_finished = torch.zeros(B if is_beam else batch_total, dtype=torch.bool, device=device)

        if forbidden_token_ids is None:
            forbidden_token_ids = (20, 21, 22, 24)
        if isinstance(forbidden_token_ids, torch.Tensor):
            forbidden_tokens = forbidden_token_ids.to(device=device, dtype=torch.long)
        else:
            forbidden_tokens = torch.tensor(list(forbidden_token_ids), device=device, dtype=torch.long)

        # ========================= [Greedy Search] =========================
        if method == "greedy_search":
            for _ in range(max_len):
                if is_finished.all(): break
                out = self.decoder(latent_exp, F.embedding(inputs, input_table).to(dtype), 
                                rope_index=torch.arange(inputs.shape[1], device=device).unsqueeze(0).expand(B, -1),
                                prefix_positional_embedding=prefix_pos_emb)
                logits = out[:, -1, :].to(torch.float32) @ output_table.T
                logits[:, forbidden_tokens] = -float('inf')
                if (inputs.shape[1]-1) < min_len: logits[:, eos_id] = -float('inf')
                
                next_token = torch.argmax(logits, dim=-1)
                next_token = torch.where(is_finished, eos_id, next_token)
                is_finished |= (next_token == eos_id)
                inputs = torch.cat([inputs, next_token.unsqueeze(-1)], dim=-1)
            return inputs

        # ========================= [Beam Search] =========================
        elif is_beam:
            beam_scores = torch.zeros((B, num_beams), device=device)
            beam_scores[:, 1:] = -1e9
            hyps = [[] for _ in range(B)]
            
            for _ in range(max_len):
                if is_finished.all(): break
                out = self.decoder(latent_exp, F.embedding(inputs, input_table).to(dtype),
                                rope_index=torch.arange(inputs.shape[1], device=device).unsqueeze(0).expand(batch_total, -1),
                                prefix_positional_embedding=prefix_pos_emb)
                
                log_probs = F.log_softmax(out[:, -1, :].to(torch.float32) @ output_table.T, dim=-1)
                log_probs[:, forbidden_tokens] = -float('inf')
                DecoderUtils.apply_eos_bias(log_probs, inputs.shape[1]-1, min_len, eos_warmup, eos_bias, eos_id)
                
                scores = beam_scores.unsqueeze(-1) + log_probs.view(B, num_beams, vocab_size)
                top_scores, top_indices = torch.topk(scores.view(B, -1), k=min(num_beams * 4, vocab_size * num_beams), dim=-1)
                
                new_beams, new_tokens, new_scores = [], [], []
                for b in range(B):
                    if is_finished[b]:

                        new_beams.extend([b*num_beams]*num_beams); new_tokens.extend([eos_id]*num_beams); new_scores.extend([-1e9]*num_beams)
                        continue

                    count = 0
                    for score, idx in zip(top_scores[b], top_indices[b]):
                        src_beam, tok = idx // vocab_size, idx % vocab_size
                        if tok == eos_id:
                            hyps[b].append({'seq': torch.cat([inputs[b*num_beams + src_beam], torch.tensor([eos_id], device=device)]), 'logp': float(score)})
                        elif count < num_beams:
                            new_beams.append(b*num_beams + src_beam); new_tokens.append(tok); new_scores.append(score)
                            count += 1
                        if count == num_beams: break
                    
                    if hyps[b]:
                        best_f = max(h['logp'] / DecoderUtils.gnmt_lp(len(h['seq'])-1) for h in hyps[b])
                        if (torch.max(beam_scores[b]) / DecoderUtils.gnmt_lp(inputs.shape[1])) <= best_f:
                            is_finished[b] = True

                inputs = torch.cat([inputs[torch.tensor(new_beams, device=device)], torch.tensor(new_tokens, device=device).unsqueeze(-1)], dim=-1)
                beam_scores = torch.tensor(new_scores, device=device).view(B, num_beams)
            
            return _format_beam_results(B, num_beams, int(num_return_sequences), hyps, inputs, beam_scores, eos_id)

        # ========================= [Sampling] =========================
        elif is_sample:
            sum_logp = torch.zeros(batch_total, device=device)
            for _ in range(max_len):
                if is_finished.all(): break
                out = self.decoder(latent_exp, F.embedding(inputs, input_table).to(dtype),
                                rope_index=torch.arange(inputs.shape[1], device=device).unsqueeze(0).expand(batch_total, -1),
                                prefix_positional_embedding=prefix_pos_emb)
                
                logits = (out[:, -1, :].to(torch.float32) @ output_table.T) / 0.9
                logits[:, forbidden_tokens] = -float('inf')
                DecoderUtils.block_no_repeat_ngram_(logits, inputs, 4)
                DecoderUtils.apply_repetition_penalty_(logits, inputs, 1.15, bos_id)
                DecoderUtils.apply_eos_bias(logits, inputs.shape[1]-1, min_len, 32, 3.0, eos_id)
                
                filtered = DecoderUtils.top_p_k_filter(logits, top_p=0.95)
                next_token = torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1).squeeze(-1)
                next_token = torch.where(is_finished, eos_id, next_token)
                
                sum_logp += F.log_softmax(filtered, dim=-1).gather(1, next_token.unsqueeze(-1)).squeeze(-1)
                is_finished |= (next_token == eos_id)
                inputs = torch.cat([inputs, next_token.unsqueeze(-1)], dim=-1)
                
            return _format_sample_results(B, group_size, inputs, sum_logp, eos_id)


        
    @torch.no_grad()
    def encode(self, inputs):
        ######### embedding #########
        self.token_codebook = self.token_codebook.to(self.protoken_embedding_dense.weight.device)
    ###512 1024
        protoken_embedding_table = self.protoken_embedding_dense(self.token_codebook)
    ###516 1024
        protoken_input_embedding_table = torch.cat(
            [protoken_embedding_table, self.common_embedding_table], dim=0
        )
        protoken_embedding = F.embedding(
            inputs['encoder_protokens'], protoken_input_embedding_table
        )
        tokens_embedding = protoken_embedding 
        tokens_embedding = tokens_embedding.to(self.arr_dtype)
        ######### encoding #########
        encode_act, prefix_positional_embedding = self.encoder(
            tokens_embedding, inputs['encoder_mask'], inputs['encoder_residue_index']
        )
        latent_act = self.encoder_down(encode_act)
        if self.global_config.task != 'evo_reason':
            latent_act = self.latent_norm_block(latent_act)
        latent_act_to_clip = latent_act.reshape(latent_act.shape[0], -1)
        latent_act_to_clip = l2_normalize(latent_act_to_clip)
        latent_clip = self.clip_projection_layer(latent_act_to_clip)
        
        return latent_act, latent_clip
    


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=64, output_dim=1,  dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.dropout(F.relu(self.fc1(z)))
        y_hat = self.fc2(h)
        return y_hat


class ProTok_lightning(L.LightningModule):
    def __init__(self, model, config, global_config, protoken_codebook=None, **kwargs):
        super().__init__()
        if protoken_codebook is None:
            placeholder = torch.zeros((22, 1280))
            self.register_buffer("protoken_codebook", placeholder)
        else:
            self.register_buffer("protoken_codebook", protoken_codebook)
        self.model = model(config, global_config, self.protoken_codebook)
        self.config = config
        self.global_config = global_config
        self.regressor = LatentRegressor(latent_dim=config.encoder.num_prefix_tokens * config.vq_config.latent_dim,
                                         hidden_dim=64, output_dim=1)

    def forward(self, x):
        
        return self.model(x)
    def encode(self, x):

        return self.model.encode(x)
    def decode_batch(self, *args, **kwargs):

        return self.model.decode_batch(*args, **kwargs)
    def predict(self, x):
        latent, clip = self.model.encode(x)
        B, num_prefix, feature_dim = latent.shape
        y_hat = self.regressor(latent.reshape(B,-1))

        return y_hat.detach().cpu().numpy().reshape(-1)
