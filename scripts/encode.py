import os
import torch
import numpy as np
import argparse
import lightning as L
from net.ProTok import ProTok_lightning
from src.common.utils import read_fasta, restype_order, build_feature
from src.common.prediction import merge_indexed_arrays
from pathlib import Path
import pickle as pkl
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size

def setup_dist():

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    else:
        return 0, 1, 0

def main():
    parser = argparse.ArgumentParser(description="ProTok Feature Extraction Tool (Distributed Supported)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/ProTok_main.ckpt", help="Path to checkpoint")
    parser.add_argument("--input", type=str, default="./example/TB2:S15435Taxa1_processed.fasta", help="Input FASTA")
    parser.add_argument("--output", type=str, default="./example/embeddings.npy", help="Output .npy path")
    parser.add_argument("--type", type=str, choices=["latent", "clip"], default="latent")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=None,
                        help="Total padded input length, including prefix and special tokens; inferred from the input if omitted.")
    parser.add_argument("--save_uncond_train_pkl", type=str, default=None)
    args = parser.parse_args()


    rank, world_size, local_rank = setup_dist()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"World size: {world_size}, Using device: {device}")
        print(f"Loading checkpoint: {args.checkpoint}")


    model = ProTok_lightning.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=False)
    model.to(device)
    model.eval()


    if is_main_process:
        print(f"Reading FASTA: {args.input}")
    names, seqs = read_fasta(args.input)
    if not seqs or len(names) != len(seqs):
        raise ValueError('Input FASTA must contain one nonempty sequence per header.')
    num_prefix = model.config.encoder.num_prefix_tokens
    required_length = max(map(len, seqs)) + num_prefix + 2
    if args.max_len is not None and args.max_len < required_length:
        raise ValueError(f'Input requires max_len >= {required_length}; increase --max_len or omit it.')
    

    features_dict = build_feature(
        seqs,
        restype_order=restype_order,
        num_prefix=num_prefix,
        max_len=args.max_len,
        mask=False,
        return_torch=True,
        torch_device="cpu",
    )


    keys = sorted(features_dict.keys())
    dataset = TensorDataset(torch.arange(len(seqs)), *(features_dict[k] for k in keys))
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=False, 
        pin_memory=True,
        num_workers=4
    )

    dim = 768 if args.type == "latent" else 128
    local_results = []
    local_indices = []

    if is_main_process:
        print(f"Extracting {args.type} features...")

    for batch_tensors in dataloader:

        local_indices.append(batch_tensors[0].numpy())
        batch = {keys[i]: batch_tensors[i + 1].to(device, non_blocking=True) for i in range(len(keys))}
        
        latent, clip = model.encode(batch)
        out_tensor = latent if args.type == "latent" else clip
        

        out_tensor = out_tensor.view(out_tensor.shape[0], -1)
        local_results.append(out_tensor.cpu().numpy())


    local_results = np.concatenate(local_results, axis=0) if local_results else np.empty((0, dim))
    local_part = {'index': np.concatenate(local_indices) if local_indices else np.empty(0, dtype=np.int64),
                  'embedding': local_results}


    if world_size > 1:
        all_results_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_results_list, local_part)
        
        if is_main_process:
            results = merge_indexed_arrays(all_results_list, len(seqs))['embedding']
    else:
        results = merge_indexed_arrays([local_part], len(seqs))['embedding']


    if is_main_process:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, results)
        print(f"Succeeded! Total shapes: {results.shape}. Saved to: {args.output}")

        if args.save_uncond_train_pkl and args.type == "latent":
            Path(args.save_uncond_train_pkl).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_uncond_train_pkl, "wb") as f:
                pkl.dump({"embedding": results.reshape(-1, num_prefix, model.config.vq_config.latent_dim),
                          "labels": np.zeros(len(results))}, f)
                print(f"Saved unconditional training dataset to: {args.save_uncond_train_pkl}")

    if world_size > 1:
        destroy_process_group()

if __name__ == "__main__":
    main()
