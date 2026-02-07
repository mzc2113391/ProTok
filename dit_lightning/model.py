from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import DiTConfig
    from .rope import apply_rotary_pos_emb
except ImportError:  # allow import when run as a script
    from config import DiTConfig
    from rope import apply_rotary_pos_emb


def get_activation(name: str):
    return {
        'relu': F.relu,
        'gelu': F.gelu,
        'silu': F.silu,
        'swish': F.silu,
    }.get(name, F.silu)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    @torch.no_grad()
    def _timestep_embedding(self, t: torch.Tensor, max_period: int = 10_000) -> torch.Tensor:
        # t: (B,) integer timesteps (not normalized)
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self._timestep_embedding(t)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x


class GaussianRBF(nn.Module):
    def __init__(self, num_basis: int, r_min: float, r_max: float, sigma: float, r_ref: float):
        super().__init__()
        self.num_basis = num_basis
        self.register_buffer('r_min', torch.tensor(r_min, dtype=torch.float32))
        self.register_buffer('r_max', torch.tensor(r_max, dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.register_buffer('r_ref', torch.tensor(r_ref, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) in [0,1]
        inv_ref = 1.0 / self.r_ref
        coeff = -0.5 / (self.sigma * self.sigma)
        r = x * inv_ref
        r = r[..., None]
        offsets = torch.linspace(self.r_min * inv_ref, self.r_max * inv_ref, self.num_basis, device=x.device)
        diff = r - offsets
        return torch.exp(coeff * diff.pow(2))


class LabelEmbedder(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_label_rbf:
            self.rbf = GaussianRBF(cfg.rbf_num_basis, cfg.rbf_r_min, cfg.rbf_r_max, cfg.rbf_sigma, cfg.rbf_r_ref)
            in_dim = cfg.rbf_num_basis
            self.proj1 = nn.Linear(in_dim, cfg.label_embed_hidden)
            self.proj2 = nn.Linear(cfg.label_embed_hidden, cfg.label_embed_hidden)
        else:
            self.embed = nn.Embedding(cfg.num_classes + 1, cfg.label_embed_hidden)  # +1 for dropout token

    def forward(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.cfg.emb_label_flag:
            return torch.zeros(labels.shape[0], self.cfg.label_embed_hidden, device=labels.device)
        if self.cfg.use_label_rbf:
            # expects labels in [0,1]
            x = self.rbf(labels)
            x = F.silu(self.proj1(x))
            x = self.proj2(x)
            if force_drop_ids is not None:
                x = torch.where(force_drop_ids[:, None].bool(), torch.zeros_like(x), x)
            else:
                if self.cfg.label_drop_rate > 0:
                    drop = torch.rand(labels.shape[0], device=labels.device) < self.cfg.label_drop_rate
                    x = torch.where(drop[:, None], torch.zeros_like(x), x)
            return x
        else:
            if force_drop_ids is None and self.cfg.label_drop_rate > 0:
                drop = torch.rand(labels.shape[0], device=labels.device) < self.cfg.label_drop_rate
            else:
                drop = force_drop_ids.bool() if force_drop_ids is not None else torch.zeros_like(labels, dtype=torch.bool)
            labels = torch.where(drop, torch.full_like(labels, self.cfg.num_classes), labels)
            return self.embed(labels)


class NormBlock(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, rope_index: Optional[torch.Tensor] = None):
        # x: (B,T,C), mask: (B,T)
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def reshape_heads(y):
            return y.view(b, t, self.heads, self.head_dim)
        q, k, v = map(reshape_heads, (q, k, v))

        if self.use_rope and rope_index is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_index)

        # attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bthd,bshd->bhts', q, k) * scale
        if mask is not None:
            # mask: 1 for valid, 0 for pad
            mask_k = mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn.masked_fill(mask_k == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.do(attn)
        y = torch.einsum('bhts,bshd->bthd', attn, v)
        y = y.contiguous().view(b, t, c)
        y = self.out(y)
        y = self.do(y)
        return y


class Transition(nn.Module):
    def __init__(self, hidden: int, factor: int = 4, dropout: float = 0.0, use_glu: bool = True, act: str = 'gelu'):
        super().__init__()
        self.use_glu = use_glu
        self.dropout = nn.Dropout(dropout)
        if use_glu:
            self.fc = nn.Linear(hidden, hidden * factor * 2)
        else:
            self.fc1 = nn.Linear(hidden, hidden * factor)
            self.fc2 = nn.Linear(hidden * factor, hidden)
        self.proj = nn.Linear(hidden * factor, hidden) if use_glu else None
        self.act = get_activation(act)

    def forward(self, x):
        if self.use_glu:
            x_proj = self.fc(x)
            a, b = x_proj.chunk(2, dim=-1)
            y = self.act(a) * b
            y = self.proj(y)
        else:
            y = self.act(self.fc1(x))
            y = self.fc2(y)
        y = self.dropout(y)
        return y


class AdaLNBlock(nn.Module):
    def __init__(self, hidden: int, module: nn.Module):
        super().__init__()
        self.hidden = hidden
        self.module = module
        self.norm = NormBlock(hidden)
        self.to_alpha_beta_gamma = nn.Linear(hidden, hidden * 3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *module_args):
        # x: (B,T,C), cond: (B,C)
        cond = F.silu(cond)
        alpha, beta, gamma = self.to_alpha_beta_gamma(cond).chunk(3, dim=-1)
        d = self.norm(x)
        d = d * (1 + gamma[:, None, :]) + beta[:, None, :]
        d = self.module(d, *module_args)
        d = d * alpha[:, None, :]
        return x + d


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig, in_features: int):
        super().__init__()
        self.cfg = cfg
        self.in_features = in_features
        self.time_embed = TimestepEmbedder(cfg.time_embed_dim, cfg.hidden_size)
        self.label_embed = LabelEmbedder(cfg) if cfg.emb_label_flag else None
        self.in_proj = nn.Linear(in_features, cfg.hidden_size, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layers):
            attn = MultiheadSelfAttention(cfg.hidden_size, cfg.n_heads, cfg.dropout, cfg.use_rope)
            trans = Transition(cfg.hidden_size, factor=cfg.mlp_factor, dropout=cfg.dropout, use_glu=cfg.use_glu)
            self.blocks.append(nn.ModuleList([
                AdaLNBlock(cfg.hidden_size, attn),
                AdaLNBlock(cfg.hidden_size, trans),
            ]))
        self.out_norm = NormBlock(cfg.hidden_size)
        self.out_mod = nn.Linear(cfg.hidden_size, in_features)
        self.out_to_alpha_beta = nn.Linear(cfg.hidden_size, cfg.hidden_size * 2)

    def forward(self,
                tokens: torch.Tensor,
                tokens_mask: torch.Tensor,
                time: torch.Tensor,
                label: Optional[torch.Tensor] = None,
                tokens_rope_index: Optional[torch.Tensor] = None,
                force_drop_ids: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # tokens: (B,T,C), mask: (B,T), time: (B,), label: (B,)
        b, t, c = tokens.shape
        time_emb = self.time_embed(time)
        label_emb = torch.zeros_like(time_emb)
        if self.label_embed is not None and label is not None:
            label_emb = self.label_embed(label, force_drop_ids)
        cond = time_emb + label_emb

        x = self.in_proj(tokens)
        for attn_blk, trans_blk in self.blocks:
            x = attn_blk(x, cond, tokens_mask, tokens_rope_index)
            x = trans_blk(x, cond)
        # Output: adaLN-style affine -> projection
        beta, gamma = self.out_to_alpha_beta(F.silu(cond)).chunk(2, dim=-1)
        y = self.out_norm(x)
        y = y * (1 + gamma[:, None, :]) + beta[:, None, :]
        y = self.out_mod(y)
        return y
