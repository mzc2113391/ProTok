from dataclasses import dataclass
from typing import Optional


@dataclass
class DiTConfig:
    # Model sizes
    hidden_size: int = 256
    n_heads: int = 16
    n_layers: int = 4  # n_iterations in JAX config
    dropout: float = 0.01
    mlp_factor: int = 4  # transition_factor
    use_glu: bool = True

    # Time and label embeds
    time_embed_dim: int = 256
    label_embed_hidden: int = 256
    emb_label_flag: bool = True
    label_drop_rate: float = 0.2
    num_classes: int = 8
    use_label_rbf: bool = False  # if True, use Gaussian RBF vs categorical embedding
    rbf_num_basis: int = 256
    rbf_r_min: float = 0.0
    rbf_r_max: float = 1.0
    rbf_sigma: float = 0.3
    rbf_r_ref: float = 1.0

    # Attention
    use_rope: bool = True


@dataclass
class TrainConfig:
    # Diffusion
    diffusion_timesteps: int = 500
    train_t_min: int = 0
    train_t_max: int = 500

    # Objective
    x_prediction: bool = False  # False: epsilon-pred; True: x0-pred
    seq_len_power: float = 0.5  # length-based weighting

    # Loss weights (x0 mode)
    w_protoken: float = 0.3
    w_protoken_cos: float = 0.0
    w_aatype: float = 0.2

    # Optim
    lr_max: float = 1e-4
    lr_min: float = 1e-5
    lr_warmup_steps: int = 2000
    lr_decay_steps: int = 2_000_000
    weight_decay: float = 1e-3
    grad_clip_norm: float = 0.1

    # Embedding tables (x0 mode)
    protoken_emb_path: Optional[str] = None
    aatype_emb_path: Optional[str] = None

