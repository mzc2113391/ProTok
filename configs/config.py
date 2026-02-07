# configs/protok_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ml_collections import ConfigDict


def CD(**kwargs) -> ConfigDict:
    """
    Compatible ConfigDict builder.
    Your ml_collections version does NOT support ConfigDict(a=1, b=2).
    Always use ConfigDict(dict(...)).
    """
    return ConfigDict(dict(kwargs))


def _maybe_lock(cfg: ConfigDict) -> None:
    # Different ml_collections versions may differ.
    if hasattr(cfg, "lock"):
        cfg.lock()  # disallow new keys
    if hasattr(cfg, "freeze"):
        # some versions support freeze (make it immutable)
        # if you don't want fully immutable, comment this out
        # cfg.freeze()
        pass

# -------------------------
# 1) Typed defaults
# -------------------------
@dataclass(frozen=True)
class ProTokDefaults:
    dim_feature: int = 768
    decoder_dim_feature: int = 768
    latent_dim_feature: int = 12

    num_head: int = 16
    num_prefix_token: int = 64

    encoder_n_layers: int = 10
    decoder_n_layers: int = 10

    dropout_rate_encoder: float = 0.01
    dropout_rate_decoder: float = 0.05
    dropout_flag: bool = True

    flash_attention_flag: bool = False
    causal_attention_flag_encoder: bool = False
    causal_attention_flag_decoder: bool = True

    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0

    uniformity_loss_weight: float = 0.5
    align_loss_weight: float = 0.75

    transition_factor: int = 2
    suffix_causal_flag: bool = False

    clip_projection_dim: int = 128
    proj_num_layers: int = 3
    clip_dropout_rate: float = 0.05
    clip_use_bias: bool = False

    bf16_flag: bool = False
    task: str = "default"  # default | evo_reason | search


# -------------------------
# 2) Small utils
# -------------------------
def _set_by_dotted_path(cfg: ConfigDict, path: str, value: Any) -> None:
    """Set cfg['a']['b']['c'] by path 'a.b.c'."""
    keys = path.split(".")
    cur: Any = cfg
    for k in keys[:-1]:
        # Create intermediate nodes if missing
        if (isinstance(cur, ConfigDict) and k not in cur) or (isinstance(cur, dict) and k not in cur):
            cur[k] = ConfigDict()
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: ConfigDict, overrides: Mapping[str, Any]) -> ConfigDict:
    """
    Supports dotted overrides on the ROOT config.
    Example:
      overrides = {"model.encoder.selfattention.n_layers": 12, "global_cfg.task": "search"}
    """
    for k, v in overrides.items():
        _set_by_dotted_path(cfg, k, v)
    return cfg


def validate(cfg: ConfigDict) -> None:
    """Fail fast: make config bugs loud."""
    if "model" not in cfg or "global_cfg" not in cfg:
        raise ValueError("compose_config() must create cfg.model and cfg.global_cfg before validate().")

    m = cfg.model
    g = cfg.global_cfg

    dim = int(m.dim_feature)
    nhead = int(m.encoder.selfattention.attention_embedding.n_head)
    if dim % nhead != 0:
        raise ValueError(f"dim_feature({dim}) must be divisible by num_head({nhead}).")

    # clip embedding dim consistency
    expected_clip_dim = int(m.encoder.num_prefix_tokens) * int(m.vq_config.latent_dim)
    got_clip_dim = int(m.clip.embedding_dim)
    if expected_clip_dim != got_clip_dim:
        raise ValueError(f"clip.embedding_dim mismatch: expected {expected_clip_dim}, got {got_clip_dim}.")

    task = str(g.task)
    if task not in ["default", "evo_reason", "search"]:
        raise ValueError(f"task not implemented: {task}")


def freeze(cfg: ConfigDict) -> ConfigDict:
    """Optional: lock to avoid accidental new keys (version-compatible)."""
    _maybe_lock(cfg)
    # If you want to recursively lock sub-configs:
    # _maybe_lock(cfg.model); _maybe_lock(cfg.global_cfg)
    return cfg


def build_model_config(d: ProTokDefaults) -> ConfigDict:
    clip_embedding_dim = d.num_prefix_token * d.latent_dim_feature

    transition_encoder = CD(
        method="gelu",
        transition_factor=d.transition_factor,
        dim_channel=d.dim_feature,
        dropout_rate=d.dropout_rate_encoder,
    )

    transition_decoder = CD(
        method="gelu",
        transition_factor=d.transition_factor,
        dim_channel=d.decoder_dim_feature,
        dropout_rate=d.dropout_rate_decoder,
    )

    attention_embedding_encoder = CD(
        attention_type="self",
        dim_feature=d.dim_feature,
        n_head=d.num_head,
        embedding_pair_flag=False,
    )

    hyper_attention_embedding_encoder = CD(kernel_type="rope", split_rope_flag=True)

    attention_kernel_encoder = CD(
        causal_flag=d.causal_attention_flag_encoder,
        flash_attention_flag=d.flash_attention_flag,
        dropout_rate=d.dropout_rate_encoder,
    )

    post_attention_encoder = CD(
        n_channel=d.dim_feature // d.num_head,
        n_head=d.num_head,
        out_dim=d.dim_feature,
        gating_flag=False,
    )

    selfattention = CD(
        attention_embedding=attention_embedding_encoder,
        hyper_attention_embedding=hyper_attention_embedding_encoder,
        attention_kernel=attention_kernel_encoder,
        post_attention=post_attention_encoder,
        hyper_attention_flag=True,
        dropout_rate=d.dropout_rate_encoder,
        n_layers=d.encoder_n_layers,
        num_prefix_tokens=d.num_prefix_token,
        dim_feature=d.dim_feature,
        transition=transition_encoder,
    )

    encoder = CD(
        selfattention=selfattention,
        num_prefix_tokens=d.num_prefix_token,
        dim_feature=d.dim_feature,
    )

    vq_config = CD(
        latent_dim=d.latent_dim_feature,
        dim_to_decoder_feature=d.decoder_dim_feature,
        entropy_loss_weight=d.entropy_loss_weight,
        diversity_gamma=d.diversity_gamma,
    )

    attention_embedding_decoder = CD(
        attention_type="self",
        dim_feature=d.decoder_dim_feature,
        n_head=d.num_head,
        embedding_pair_flag=False,
    )

    hyper_attention_embedding_decoder = CD(kernel_type="rope", split_rope_flag=True)

    attention_kernel_decoder = CD(
        causal_flag=d.causal_attention_flag_decoder,
        flash_attention_flag=d.flash_attention_flag,
        dropout_rate=d.dropout_rate_decoder,
    )

    post_attention_decoder = CD(
        n_channel=d.decoder_dim_feature // d.num_head,
        n_head=d.num_head,
        out_dim=d.decoder_dim_feature,
        gating_flag=False,
    )

    attention_decoder = CD(
        attention_embedding=attention_embedding_decoder,
        hyper_attention_embedding=hyper_attention_embedding_decoder,
        attention_kernel=attention_kernel_decoder,
        post_attention=post_attention_decoder,
        hyper_attention_flag=True,
        dropout_rate=d.dropout_rate_decoder,
    )

    adaLN = CD(hidden_size=d.decoder_dim_feature)

    causalattention = CD(
        attention=attention_decoder,
        transition=transition_decoder,
        num_prefix_tokens=d.num_prefix_token,
        dim_feature=d.decoder_dim_feature,
        adaLN=adaLN,
        hyper_attention_flag=True,
        dropout_rate=d.dropout_rate_decoder,
        n_layers=d.decoder_n_layers,
    )

    decoder = CD(
        causalattention=causalattention,
        latent_dim=d.latent_dim_feature,
        dim_feature=d.decoder_dim_feature,
        num_prefix_tokens=d.num_prefix_token,
        attention_factor_head=d.dim_feature // d.latent_dim_feature,
    )

    contrastive = CD(
        t=2,
        alpha=2,
        uniformity_loss_weight=d.uniformity_loss_weight,
        align_loss_weight=d.align_loss_weight,
    )

    clip_cfg = CD(
        projection_dim=d.clip_projection_dim,
        embedding_dim=clip_embedding_dim,
        dropout_rate=d.clip_dropout_rate,
        proj_num_layers=d.proj_num_layers,
        activate_final=False,
        dropout_flag=d.dropout_flag,
        use_bias=d.clip_use_bias,
    )

    model_cfg = CD(
        encoder=encoder,
        vq_config=vq_config,
        decoder=decoder,
        dim_feature=d.dim_feature,
        protoken_vocabsize=512,
        contrastive=contrastive,
        attention_factor_head=d.dim_feature // d.latent_dim_feature,
        clip=clip_cfg,
    )

    return model_cfg


def build_global_config(d: ProTokDefaults) -> ConfigDict:
    return CD(
        bf16_flag=d.bf16_flag,
        dropout_flag=d.dropout_flag,
        norm_method="layernorm",
        norm_small=1e-6,
        suffix_causal_flag=d.suffix_causal_flag,
        task=d.task,
    )


def compose_config(
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    defaults: ProTokDefaults = ProTokDefaults(),
    freeze_cfg: bool = True,
) -> ConfigDict:
    """
    Returns a single root config:
      cfg.model      -> (old ProTok_config)
      cfg.global_cfg -> (old global_config)
    """
    cfg = ConfigDict()
    cfg.model = build_model_config(defaults)
    cfg.global_cfg = build_global_config(defaults)

    # dotted overrides on ROOT
    if overrides:
        apply_overrides(cfg, overrides)

    validate(cfg)
    if freeze_cfg:
        freeze(cfg)

    return cfg


def get_default_config() -> ConfigDict:
    return compose_config()
