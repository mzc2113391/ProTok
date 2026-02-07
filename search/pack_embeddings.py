#!/usr/bin/env python3
"""
Convert .npy embeddings to a raw float16/float32 memmap file for fast IO.

Output is a headerless binary file with shape (N, D).
A sidecar JSON meta file can be written with shape/dtype info.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pack .npy embeddings into raw memmap format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input .npy embeddings")
    p.add_argument("--output", required=True, help="Output raw memmap path (.fbin/.mmap)")
    p.add_argument("--dtype", default="float16", help="Target dtype (float16 or float32)")
    p.add_argument("--batch-size", type=int, default=1_000_000, help="Conversion batch size")
    p.add_argument("--meta", default=None, help="Optional meta JSON path")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    emb = np.load(in_path, mmap_mode="r")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")

    N, d = emb.shape
    target_dtype = np.dtype(args.dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = np.memmap(out_path, dtype=target_dtype, mode="w+", shape=(N, d))

    bs = int(args.batch_size)
    for i in range(0, N, bs):
        chunk = np.asarray(emb[i : i + bs], dtype=target_dtype)
        out[i : i + chunk.shape[0]] = chunk

    out.flush()

    meta = {
        "input": str(in_path),
        "output": str(out_path),
        "count": int(N),
        "dim": int(d),
        "dtype": str(target_dtype),
    }
    meta_path = Path(args.meta) if args.meta else Path(str(out_path) + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Packed embeddings: N={N}, d={d}, dtype={target_dtype}")
    print(f"Output: {out_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
