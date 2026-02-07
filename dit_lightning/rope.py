import math
import torch


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
    """
    Apply RoPE to q and k.
    q, k: (B, T, H, D)
    pos: (B, T) integer position index
    """
    b, t, h, d = q.shape
    assert d % 2 == 0, "Head dimension must be even for RoPE"
    # Build sin/cos cache per each position value
    # Use global max position in batch to build angles
    max_pos = int(pos.max().item()) if pos.numel() > 0 else t - 1
    device = q.device
    dtype = q.dtype
    half = d // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(0, half, device=device, dtype=dtype) / half)
    # angles: (P, H) independent of head; broadcast later
    all_idx = torch.arange(0, max_pos + 1, device=device, dtype=dtype)
    angles = torch.einsum('p,h->ph', all_idx, freqs)
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    # gather batch positions
    pos_clamped = pos.clamp(min=0, max=max_pos)
    sin_pos = sin[pos_clamped]  # (B,T,half)
    cos_pos = cos[pos_clamped]  # (B,T,half)

    def rope_rotate(x):
        # x: (B,T,H,D)
        x1, x2 = x[..., :half], x[..., half:]
        # broadcast sin/cos to heads
        sin_b = sin_pos[..., None, :]
        cos_b = cos_pos[..., None, :]
        x1_rot = x1 * cos_b - x2 * sin_b
        x2_rot = x1 * sin_b + x2 * cos_b
        return torch.cat([x1_rot, x2_rot], dim=-1)

    return rope_rotate(q), rope_rotate(k)

