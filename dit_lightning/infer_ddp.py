import argparse
import math
import os
import pickle as pkl
from typing import List

import torch
import torch.distributed as dist
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from .lightning_module import DiTLightning
except ImportError:
    from lightning_module import DiTLightning


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--out_pkl', type=str, required=True)
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--samples_per_class', type=int, default=100)
    p.add_argument('--samples_per_device', type=int, default=256, help='Samples per process (GPU) per outer step')
    p.add_argument('--timesteps', type=int, default=500, help='If set, cap diffusion steps')
    p.add_argument('--eq_step', type=int, default=20)
    p.add_argument('--cfg_weight', type=float, default=0.5)
    p.add_argument('--no_cfg', action='store_true', help='Disable classifier-free guidance; use conditional only')
    p.add_argument('--uncond', action='store_true', help='Unconditional sampling (drop labels)')
    p.add_argument('--clip_x0', action='store_true', help='Clamp predicted x0 to [-1,1]')
    p.add_argument('--seed', type=int, default=114)
    p.add_argument('--labels', type=str, default=None, help='Comma-separated list of labels to generate (e.g., "6,7"). If not set, generates for all classes.')
    p.add_argument('--local_rank', type=int, default=None)
    p.add_argument('--rank', type=int, default=None)
    p.add_argument('--world_size', type=int, default=None)
    return p.parse_args()


def ddp_init(args):
    if args.rank is None:
        args.rank = int(os.environ.get('RANK', '0'))
    if args.local_rank is None:
        args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.world_size is None:
        args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def partition_indices(total: int, world_size: int, rank: int):
    per = math.ceil(total / world_size)
    start = rank * per
    end = min(start + per, total)
    return start, end


@torch.no_grad()
def main():
    torch.set_float32_matmul_precision("high")
     
    args = parse_args()
    ddp_init(args)
    device = torch.device(f'cuda:{args.local_rank}')
    torch.manual_seed(args.seed + args.rank)

    # Load module
    lit: DiTLightning = DiTLightning.load_from_checkpoint(args.ckpt_path, map_location=device)
    lit.eval()
    lit.to(device)
    model = lit.model
    model.eval().to(device)
    # Wrap with DDP (optional for inference, but provides symmetry and avoids device 0 bottleneck)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

    diffusion = lit.diffusion
    feat_dim = lit.in_features if getattr(lit, 'in_features', None) is not None else 12
    num_classes_ckpt = int(getattr(getattr(lit, 'dit_cfg', None), 'num_classes', 8))

    # Steps
    T_total = int(getattr(diffusion, 'num_timesteps', 1000))
    if args.timesteps is not None:
        T_total = min(T_total, int(args.timesteps))

    # Global target count handling
    samples_per_class = int(args.samples_per_class)

    if args.labels is not None and len(args.labels.strip()) > 0:
        try:
            target_labels_list = [int(l) for l in args.labels.split(',')]
        except ValueError:
            raise ValueError(f"Invalid format for --labels: {args.labels}. Expected comma-separated integers.")
    else:

        target_labels_list = list(range(num_classes_ckpt))


    NDATA = len(target_labels_list) * samples_per_class

    target_labels_tensor = torch.tensor(target_labels_list, device=device, dtype=torch.long)
    labels_all = target_labels_tensor.repeat_interleave(samples_per_class)

    # Local partition
    g_start, g_end = partition_indices(NDATA, args.world_size, args.rank)
    local_N = max(0, g_end - g_start)

    # Slice locally
    labels = labels_all[g_start:g_end]

    # Chunk locally by memory
    per_step = int(args.samples_per_device)
    n_steps = math.ceil(local_N / per_step) if local_N > 0 else 0

    local_embeddings = []
    local_seq_masks = []
    local_res_idx = []
    local_labels = []

    # Optional progress bars (only on rank 0 to avoid clutter)
    step_iter = range(n_steps)
    step_pbar = None
    if args.rank == 0 and tqdm is not None:
        step_pbar = tqdm(total=n_steps, desc='DDP Steps', position=0)

    for step in step_iter:
        s = step * per_step
        e = min((step + 1) * per_step, local_N)
        bs = e - s
        if bs <= 0:
            continue
        labels_b = labels[s:e]

        seq_len = int(args.seq_len)
        seq_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)
        residue_index = torch.arange(seq_len, device=device).repeat(bs, 1)
        x = torch.randn(bs, seq_len, feat_dim, device=device)

        def denoise_step(x_t, t_scalar):
            t = torch.full((x_t.shape[0],), int(t_scalar), device=device) 
            drop_cond = torch.zeros_like(labels_b)
            if args.uncond:
                drop_uncond = torch.ones_like(labels_b)
                eps_guided = ddp_model(x_t, seq_mask, t, label=labels_b, tokens_rope_index=residue_index, force_drop_ids=drop_uncond)
            elif args.no_cfg:
                eps_guided = ddp_model(x_t, seq_mask, t, label=labels_b, tokens_rope_index=residue_index, force_drop_ids=drop_cond)
            else:
                drop_uncond = torch.ones_like(labels_b)
                eps_cond = ddp_model(x_t, seq_mask, t, label=labels_b, tokens_rope_index=residue_index, force_drop_ids=drop_cond)
                eps_uncond = ddp_model(x_t, seq_mask, t, label=labels_b, tokens_rope_index=residue_index, force_drop_ids=drop_uncond)
                eps_guided = (1 + args.cfg_weight) * eps_cond - args.cfg_weight * eps_uncond
            mean, _, logvar = diffusion.p_mean_variance(x_t, t, eps_guided, x_prediction=False, clip=bool(args.clip_x0))
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.exp(0.5 * logvar) * noise
            return x_prev

        # Sampling loop 
        time_iter = range(T_total - 1, -1, -1)
        if args.rank == 0 and tqdm is not None:
            time_iter = tqdm(time_iter, desc=f'Timesteps (step {step+1}/{n_steps})', position=1, leave=False)
        for ti in time_iter:
            for _ in range(max(0, int(args.eq_step))):
                x = denoise_step(x, ti)
                t = torch.full((bs,), int(ti), device=device)
                x = diffusion.q_sample_step(x, t, torch.randn_like(x))
            x = denoise_step(x, ti)

        if step_pbar is not None:
            step_pbar.update(1)

        local_embeddings.append(x.detach().cpu())
        local_seq_masks.append(seq_mask.detach().cpu())
        local_res_idx.append(residue_index.detach().cpu())
        local_labels.append(labels_b.detach().cpu())

    # Convert to numpy and gather
    embed_np = torch.cat(local_embeddings, dim=0).numpy() if local_embeddings else None
    mask_np = torch.cat(local_seq_masks, dim=0).numpy() if local_seq_masks else None
    resi_np = torch.cat(local_res_idx, dim=0).numpy() if local_res_idx else None
    label_np = torch.cat(local_labels, dim=0).numpy() if local_labels else None

    payload = {'embedding': embed_np, 'seq_mask': mask_np, 'residue_index': resi_np, 'label': label_np, 'start': g_start, 'end': g_end}

    gathered = None
    if args.rank == 0:
        if step_pbar is not None:
            step_pbar.close()
        gathered = [None for _ in range(args.world_size)]
    dist.gather_object(payload, gathered, dst=0)

    if args.rank == 0:
        # Assemble in global order
        embedding_list = [torch.from_numpy(p['embedding']) for p in gathered if p['embedding'] is not None]
        seq_mask_list = [torch.from_numpy(p['seq_mask']) for p in gathered if p['seq_mask'] is not None]
        res_idx_list = [torch.from_numpy(p['residue_index']) for p in gathered if p['residue_index'] is not None]
        label_list = [torch.from_numpy(p['label']) for p in gathered if p['label'] is not None]
    
        embedding = torch.cat(embedding_list, dim=0).numpy() if embedding_list else None
        seq_mask = torch.cat(seq_mask_list, dim=0).numpy() if seq_mask_list else None
        residue_index = torch.cat(res_idx_list, dim=0).numpy() if res_idx_list else None
        label = torch.cat(label_list, dim=0).numpy() if label_list else None

        # Trim to NDATA (safety)
        if embedding is not None:
            embedding = embedding[:NDATA]
            seq_mask = seq_mask[:NDATA]
            residue_index = residue_index[:NDATA]
            label = label[:NDATA]

        os.makedirs(os.path.dirname(args.out_pkl) or '.', exist_ok=True)
        with open(args.out_pkl, 'wb') as f:
            pkl.dump({'embedding': embedding, 'seq_mask': seq_mask, 'residue_index': residue_index, 'label': label}, f)
        print(f'[Rank 0] Saved {embedding.shape[0] if embedding is not None else 0} samples to {args.out_pkl}')

    ddp_cleanup()


if __name__ == '__main__':
    main()