import os
import sys
import argparse
import pickle as pkl
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from net.ProTok import ProTok_lightning
from src.common.utils import restypes, order_restype, create_feature_list, decode_to_seq
import lightning as L

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, dtype=torch.float32):
        """
        embeddings: [N, ...]
        """
        self.emb = embeddings
        self.dtype = dtype

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, idx: int):
        x = torch.tensor(self.emb[idx], dtype=self.dtype)
        return idx, x

def collate_fn(batch: List[Tuple[int, torch.Tensor]]):
    idxs, xs = zip(*batch)
    xs = torch.stack(xs, dim=0)  # [B, ...]
    return torch.tensor(idxs, dtype=torch.long), xs


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        os.environ.setdefault("LOCAL_RANK", "0")

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_ddp():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)


def _finalize_and_dump(all_parts: List[List[Tuple[int, str]]], N: int, out_path: str):

    merged_first: Dict[int, str] = {}
    for part in all_parts:
        if not part:
            continue
        for idx, seq in part:
            if idx not in merged_first:
                merged_first[idx] = seq  

    missing = [i for i in range(N) if i not in merged_first]
    if missing:
        raise RuntimeError(f"Missing indices after gather (got {len(merged_first)}/{N}). "
                           f"Examples of missing: {missing[:10]}")

    result_seq_list = [merged_first[i] for i in range(N)]

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(result_seq_list, f)

    print(f"[OK] Saved {out_path}   (#seq={len(result_seq_list)})")

def main():
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-pkl", type=str, required=True)
    parser.add_argument("--output-pkl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--method", type=str, default="beam_search", choices=["beam_search", "greedy_search", "top_p"])
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument("--min-len", type=int, default=150)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--cond", action="store_true")
    parser.add_argument("--nrs", type=int, default=1)  

    args = parser.parse_args()

    ddp_init()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model: ProTok_lightning = ProTok_lightning.load_from_checkpoint(
        args.checkpoint,
        strict = False
    )

    model.eval().to(device)

    torch.set_grad_enabled(False)

    with open(args.input_pkl, "rb") as f:
        embedding_data = pkl.load(f)

    embeddings = np.asarray(embedding_data["embedding"])

    N = embeddings.shape[0]

    dataset = EmbeddingDataset(embeddings)
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    local_results: List[Tuple[int, str]] = []

    if sampler is not None:
        sampler.set_epoch(0) 

    iterable = loader
    if is_main_process():
        iterable = tqdm(loader, desc=f"Infer@rank0", ncols=100)

    for idxs, batch in iterable:
        batch = batch.to(device, non_blocking=True)
        beam_search_result = model.decode_batch(
            batch,
            method=args.method,
            max_len=args.max_len,
            min_len=args.min_len,
            num_beams=args.num_beams,
            num_return_sequences=args.nrs,
        )
        if args.method == "beam_search":
            for i, one in enumerate(beam_search_result):
                best_seq_tensor = one["best_sequence"]  # Tensor of token ids
                seq_str = decode_to_seq(best_seq_tensor)
                local_results.append((int(idxs[i]), seq_str))
        elif args.method == "greedy_search":
            for i, one in enumerate(beam_search_result):
                best_seq_tensor = one  # Tensor of token ids
                seq_str = decode_to_seq(best_seq_tensor)
                local_results.append((int(idxs[i]), seq_str))
        else:
            for i, one in enumerate(beam_search_result):
                best_seq_tensor = one["best_sequence"]
                seq_str = decode_to_seq(best_seq_tensor)
                local_results.append((int(idxs[i]), seq_str))

    # -------------------- Gather & Save --------------------
    if is_ddp():
        world_size = dist.get_world_size()
        gather_list = [None for _ in range(world_size)]
        dist.all_gather_object(gather_list, local_results)

        if is_main_process():
            _finalize_and_dump(gather_list, N, args.output_pkl)

        dist.barrier()

        if args.cond and (not is_main_process()):
            ddp_cleanup()
            sys.exit(0)
        ddp_cleanup()
    else:
        _finalize_and_dump([local_results], N, args.output_pkl)

    # -------------------- Cond: Predict fitness (rank0 only) --------------------
    if args.cond:
        print("--------------------Predicting fitness for generated sequences--------------------")

        pred_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model: ProTok_lightning = ProTok_lightning.load_from_checkpoint(
            args.checkpoint,
            strict=False
        )
        model.eval().to(pred_device)

        with open(args.output_pkl, "rb") as f:
            gen_seqs = pkl.load(f)  # list[str]

        fitness_list = []
        data_list = create_feature_list(gen_seqs, device=pred_device)

        with torch.no_grad():
            for data_dict in data_list:
                y_hat = model.predict(data_dict)
                fitness_list.append(float(y_hat[0]))

        with open(args.output_pkl, "wb") as f:
            pkl.dump({"seqs": gen_seqs, "fitness": fitness_list}, f)

        print(f"[OK] Saved with predicted fitness {args.output_pkl}   (#seq={len(gen_seqs)})")

if __name__ == "__main__":
    main()