#!/usr/bin/env python3
"""
build_faiss_index.py

Build FAISS indexes from embeddings stored as:
  - np.ndarray / np.memmap objects (N and d from .shape)
  - .npy files (supports mmap_mode="r"; N and d from the header)
  - .npz files (N and d from the stored array)
  - raw headerless memmap files (requires --dim; N inferred from file size)

Index types:
  - ivfpq: OPQ + IVFPQ (approximate)
  - flatip: IndexFlatIP (exact brute force)

Similarity:
  - Inner Product (IP) by default
  - If --normalize is set, vectors are L2-normalized and IP becomes cosine similarity
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import faiss


ArrayLike = Union[np.ndarray, np.memmap, Sequence[Sequence[float]]]


# -----------------------------
# Logging
# -----------------------------
def setup_logger(verbosity: int) -> logging.Logger:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("build_faiss_index")


# -----------------------------
# Embeddings loading
# -----------------------------
def as_2d_array(emb: ArrayLike) -> Union[np.ndarray, np.memmap]:
    """
    Normalize different embedding containers into a 2D array-like object.
    If emb is a Python list, it will be converted into a np.ndarray (in memory).
    """
    if isinstance(emb, (np.ndarray, np.memmap)):
        if emb.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
        return emb

    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {arr.shape}")
    return arr


def load_embeddings_from_path(
    path: Union[str, Path],
    *,
    dtype: str = "float16",
    npz_key: Optional[str] = None,
    dim: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Union[np.ndarray, np.memmap]:
    """
    Load embeddings from:
      - .npy  (shape inferred from header; supports mmap)
      - .npz  (shape inferred from stored array; loads into memory)
      - raw headerless memmap file (requires --dim; N inferred from file size)

    Returns an array-like with shape (N, d).
    """
    logger = logger or logging.getLogger("build_faiss_index")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings in {path}, got shape {arr.shape}")
        logger.info(f"Loaded .npy embeddings (mmap) shape={arr.shape}, dtype={arr.dtype}")
        return arr

    if suffix == ".npz":
        with np.load(path) as z:
            if npz_key is None:
                if len(z.files) != 1:
                    raise ValueError(f"{path} contains multiple arrays {z.files}; specify --npz-key.")
                npz_key = z.files[0]
            arr = z[npz_key]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings in {path}, got shape {arr.shape}")
        logger.info(f"Loaded .npz embeddings shape={arr.shape}, dtype={arr.dtype}, key={npz_key}")
        return arr

    # Raw headerless memmap
    if dim is None:
        raise ValueError(
            "Raw memmap input has no header, so d cannot be inferred. "
            "Provide --dim, and N will be inferred from file size."
        )

    dt = np.dtype(dtype)
    d = int(dim)
    byte_size = path.stat().st_size
    itemsize = dt.itemsize

    denom = itemsize * d
    if byte_size % denom != 0:
        raise ValueError(
            f"File size {byte_size} is not divisible by dtype_size({itemsize}) * d({d}). "
            "Check --dtype/--dim."
        )
    N = byte_size // denom
    arr = np.memmap(path, dtype=dt, mode="r", shape=(N, d))
    logger.info(f"Loaded raw memmap embeddings shape={arr.shape}, dtype={arr.dtype}")
    return arr


# -----------------------------
# Batch helpers
# -----------------------------
def l2_normalize_inplace(x: np.ndarray, eps: float = 1e-12) -> None:
    # Safe L2 normalization in numpy (no FAISS call)
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, eps)
    x /= norms


def iter_batches(
    emb: Union[np.ndarray, np.memmap],
    *,
    batch_size: int,
    normalize: bool,
) -> Iterator[np.ndarray]:
    """
    Yield contiguous float32 batches, optionally L2-normalized.
    """
    N = emb.shape[0]
    for i in range(0, N, batch_size):
        xb = np.asarray(emb[i : i + batch_size], dtype=np.float32)
        xb = np.ascontiguousarray(xb)
        if normalize:
            if not xb.flags.writeable:
                xb = xb.copy()
            l2_normalize_inplace(xb)
        yield xb


def sample_training_matrix(
    emb: Union[np.ndarray, np.memmap],
    *,
    train_size: int,
    seed: int,
    pack_size: int,
    normalize: bool,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Sample a training matrix (float32) from embeddings.
    For memmap, indices are sorted to reduce random I/O.
    """
    N, d = emb.shape
    train_size = int(min(train_size, N))

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=train_size, replace=False)
    idx.sort()

    train = np.empty((train_size, d), dtype=np.float32)
    pos = 0

    logger.info(f"Preparing training matrix: train_size={train_size}, d={d}, pack_size={pack_size}")
    for j in range(0, train_size, pack_size):
        sel = idx[j : j + pack_size]
        xb = np.asarray(emb[sel], dtype=np.float32)
        xb = np.ascontiguousarray(xb)
        if normalize:
            l2_normalize_inplace(xb)
        train[pos : pos + len(sel)] = xb
        pos += len(sel)

    return train


# -----------------------------
# Index builders
# -----------------------------
def build_flat_ip_index(d: int) -> faiss.Index:
    return faiss.IndexFlatIP(d)


def build_opq_ivfpq_ip_index(
    d: int,
    *,
    nlist: int,
    m: int,
    nbits: int,
    use_opq: bool,
) -> faiss.Index:
    """
    Build an OPQ + IVFPQ index with inner product metric.
    """
    if d % m != 0:
        raise ValueError(f"m ({m}) must divide d ({d}).")

    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(
        quantizer, d, int(nlist), int(m), int(nbits), faiss.METRIC_INNER_PRODUCT
    )
    if not use_opq:
        return ivfpq

    opq = faiss.OPQMatrix(d, m)
    return faiss.IndexPreTransform(opq, ivfpq)


def maybe_to_gpu(
    index: faiss.Index,
    *,
    use_gpu: bool,
    logger: logging.Logger,
) -> Tuple[faiss.Index, bool]:
    if not use_gpu:
        return index, False
    try:
        logger.info("Attempting to move index to GPU(s)...")
        index = faiss.index_cpu_to_all_gpus(index)
        logger.info("GPU index enabled.")
        return index, True
    except Exception as e:
        logger.warning(f"GPU unavailable or faiss-gpu not installed: {e}")
        logger.info("Falling back to CPU index.")
        return index, False


def maybe_to_cpu(index: faiss.Index, *, on_gpu: bool, logger: logging.Logger) -> faiss.Index:
    if not on_gpu:
        return index
    logger.info("Moving index back to CPU for serialization...")
    return faiss.index_gpu_to_cpu(index)


# -----------------------------
# Main build routine
# -----------------------------
def build_index(
    emb: ArrayLike,
    *,
    index_type: str,
    output_path: Union[str, Path],
    use_gpu: bool,
    add_batch_size: int,
    normalize: bool,
    # IVFPQ params
    nlist: int,
    m: int,
    nbits: int,
    train_size: int,
    train_seed: int,
    train_pack_size: int,
    use_opq: bool,
    min_train_per_centroid: int,
    logger: logging.Logger,
) -> None:
    emb2 = as_2d_array(emb)
    N, d = emb2.shape
    output_path = Path(output_path)

    logger.info(f"Embeddings: N={N}, d={d}, dtype={emb2.dtype}")
    logger.info(f"Index type: {index_type}")
    logger.info(f"Metric: IP{' + L2-normalization (cosine)' if normalize else ''}")

    if index_type == "flatip":
        index = build_flat_ip_index(d)
        index, on_gpu = maybe_to_gpu(index, use_gpu=use_gpu, logger=logger)

        logger.info(f"Adding vectors (batch_size={add_batch_size})...")
        added = 0
        for xb in iter_batches(emb2, batch_size=add_batch_size, normalize=normalize):
            index.add(xb)
            added += xb.shape[0]
            logger.info(f"  added {added}/{N}")

        index = maybe_to_cpu(index, on_gpu=on_gpu, logger=logger)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        logger.info(f"Saved index: {output_path}")
        return

    if index_type == "ivfpq":
        need = int(min_train_per_centroid) * int(nlist)
        eff_train_size = int(min(max(train_size, need), N))

        logger.info(f"IVFPQ params: nlist={nlist}, m={m}, nbits={nbits}, use_opq={use_opq}")
        logger.info(f"Training size: requested={train_size}, heuristic_min={need}, effective={eff_train_size}")

        index = build_opq_ivfpq_ip_index(d, nlist=nlist, m=m, nbits=nbits, use_opq=use_opq)
        index, on_gpu = maybe_to_gpu(index, use_gpu=use_gpu, logger=logger)

        train = sample_training_matrix(
            emb2,
            train_size=eff_train_size,
            seed=train_seed,
            pack_size=train_pack_size,
            normalize=normalize,
            logger=logger,
        )

        logger.info("Training index...")
        index.train(train)
        logger.info("Training done.")

        logger.info(f"Adding vectors (batch_size={add_batch_size})...")
        added = 0
        for xb in iter_batches(emb2, batch_size=add_batch_size, normalize=normalize):
            index.add(xb)
            added += xb.shape[0]
            if (added // add_batch_size) % 10 == 0:
                logger.info(f"  added {added}/{N}")

        index = maybe_to_cpu(index, on_gpu=on_gpu, logger=logger)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        logger.info(f"Saved index: {output_path}")

        logger.info("Search hint: set nprobe (e.g., 128~256) to balance speed/recall.")
        logger.info("Two-stage hint: coarse topK -> rerank with exact cosine on original vectors.")
        return

    raise ValueError(f"Unknown index_type: {index_type} (expected: ivfpq or flatip)")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FAISS indexes (OPQ+IVFPQ or FlatIP) from embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--embeddings", type=str, required=True, help="Embeddings file path (.npy/.npz/raw memmap).")
    p.add_argument("--output", type=str, required=True, help="Output FAISS index path.")
    p.add_argument("--index-type", choices=["ivfpq", "flatip"], required=True)

    # Input options
    p.add_argument("--dtype", type=str, default="float16", help="dtype for raw memmap input.")
    p.add_argument("--dim", type=int, default=128, help="Embedding dimension for raw memmap input (required for raw).")
    p.add_argument("--npz-key", type=str, default=None, help="Array key for .npz input.")

    # Similarity options
    p.add_argument("--normalize", action="store_true", help="L2-normalize vectors (IP becomes cosine).")

    # Performance options
    p.add_argument("--gpu", action="store_true", help="Use GPU(s) if available.")
    p.add_argument("--add-batch-size", type=int, default=1_000_000, help="Batch size when adding vectors.")
    p.add_argument("-v", "--verbose", action="count", default=0)

    # IVFPQ options
    p.add_argument("--nlist", type=int, default=65536, help="Number of IVF centroids (ivfpq only).")
    p.add_argument("--m", type=int, default=64, help="Number of PQ subquantizers (ivfpq only).")
    p.add_argument("--nbits", type=int, default=8, help="Bits per subquantizer code (ivfpq only).")
    p.add_argument("--train-size", type=int, default=6_000_000, help="Training sample size (ivfpq only).")
    p.add_argument("--train-seed", type=int, default=7272, help="RNG seed for training sampling (ivfpq only).")
    p.add_argument("--train-pack-size", type=int, default=2_000_000, help="Chunk size for training matrix (ivfpq only).")
    p.add_argument("--no-opq", action="store_true", help="Disable OPQ pretransform (ivfpq only).")
    p.add_argument("--min-train-per-centroid", type=int, default=39, help="Heuristic min train samples per centroid.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    emb = load_embeddings_from_path(
        args.embeddings,
        dtype=args.dtype,
        npz_key=args.npz_key,
        dim=args.dim,
        logger=logger,
    )

    build_index(
        emb,
        index_type=args.index_type,
        output_path=args.output,
        use_gpu=args.gpu,
        add_batch_size=args.add_batch_size,
        normalize=args.normalize,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        train_size=args.train_size,
        train_seed=args.train_seed,
        train_pack_size=args.train_pack_size,
        use_opq=(not args.no_opq),
        min_train_per_centroid=args.min_train_per_centroid,
        logger=logger,
    )


if __name__ == "__main__":
    main()
