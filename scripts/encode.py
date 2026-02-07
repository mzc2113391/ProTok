import os
import torch
import numpy as np
import argparse
import lightning as L
from net.ProTok import ProTok_lightning
from src.common.utils import read_fasta, restype_order, build_feature
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
    

    features_dict = build_feature(
        seqs,
        restype_order=restype_order,
        num_prefix=64,
        max_len=1024,
        mask=False,
        return_torch=True,
        torch_device="cpu",
    )


    keys = sorted(features_dict.keys())
    dataset = TensorDataset(*(features_dict[k] for k in keys))
    
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

    if is_main_process:
        print(f"Extracting {args.type} features...")

    for batch_tensors in dataloader:

        batch = {keys[i]: batch_tensors[i].to(device, non_blocking=True) for i in range(len(keys))}
        
        latent, clip = model.encode(batch)
        out_tensor = latent if args.type == "latent" else clip
        

        out_tensor = out_tensor.view(out_tensor.shape[0], -1)
        local_results.append(out_tensor.cpu().numpy())


    local_results = np.concatenate(local_results, axis=0) if local_results else np.empty((0, dim))


    if world_size > 1:
        # 收集所有进程的结果
        all_results_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(all_results_list, local_results)
        
        if is_main_process:
            # 合并结果，注意：DistributedSampler 可能会为了补齐 batch 对数据进行 padding
            results = np.concatenate(all_results_list, axis=0)
            results = results[:len(seqs)]  # 裁剪掉多余的 padding 部分
    else:
        results = local_results


    if is_main_process:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.save(args.output, results)
        print(f"Succeeded! Total shapes: {results.shape}. Saved to: {args.output}")

        if args.save_uncond_train_pkl and args.type == "latent":
            with open(args.save_uncond_train_pkl, "wb") as f:
                pkl.dump({"embedding": results.reshape(-1, 64, 12), "labels": np.zeros(len(results))}, f)
                print(f"Saved unconditional training dataset to: {args.save_uncond_train_pkl}")

    if world_size > 1:
        destroy_process_group()

if __name__ == "__main__":
    main()