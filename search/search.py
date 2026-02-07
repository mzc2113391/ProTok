#!/usr/bin/env python3
"""
Unified FAISS search script (FlatIP / IVF* / IVFPQ / etc.)

Features:
- Loads query embeddings from:
  1) .npy (N x D or D)
  2) .pkl containing either:
     - a list of embeddings (each is length D)
     - a numpy array (N x D or D)
- Loads a FAISS index from a given path
- Automatically clamps top-k to min(requested_k, index.ntotal)
- Automatically chooses a query batch size as min(num_queries, 1000) unless overridden
- Optionally sets nprobe for IVF-based indices (ignored for Flat indices)
- Saves results to .npz: {indices: int64 [N,K], distances: float32 [N,K]}
- Optional L2-normalization for cosine similarity
- Optional TSV output with name mapping (via .names.txt/.names.idx)
- Optional exact rerank using original DB embeddings
"""

import argparse
import os
import sys
from typing import Any, Optional, Tuple

import faiss
import numpy as np
import pickle as pkl


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_queries(path: str, mmap_npy: bool = True) -> np.ndarray:
    """
    Load query embeddings from .npy or .pkl.
    Returns a float32 numpy array of shape [N, D].
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Query file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, mmap_mode="r" if mmap_npy else None)
        q = np.asarray(arr)
    elif ext == ".pkl":
        with open(path, "rb") as f:
            obj = pkl.load(f)

        if isinstance(obj, np.ndarray):
            q = obj
        elif isinstance(obj, list):
            if len(obj) == 0:
                raise ValueError("Pickle list is empty.")
            q = np.asarray(obj)
        else:
            raise TypeError(
                f"Unsupported pickle content type: {type(obj)}. "
                "Expected numpy array or list of embeddings."
            )
    else:
        raise ValueError(f"Unsupported query file extension: {ext}. Use .npy or .pkl")

    q = np.asarray(q)

    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2:
        raise ValueError(f"Query array must be 2D (N x D). Got shape: {q.shape}")

    # Ensure float32 and contiguous for FAISS
    if q.dtype != np.float32:
        q = q.astype(np.float32, copy=False)
    if not q.flags["C_CONTIGUOUS"]:
        q = np.ascontiguousarray(q)

    # Basic sanity checks
    if q.shape[0] <= 0 or q.shape[1] <= 0:
        raise ValueError(f"Invalid query shape: {q.shape}")

    return q


def l2_normalize_inplace(x: np.ndarray, eps: float = 1e-12) -> None:
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    norms = np.maximum(norms, eps)
    x /= norms


def load_embeddings_from_path(
    path: str,
    *,
    dtype: str = "float16",
    npz_key: Optional[str] = None,
    dim: Optional[int] = None,
) -> np.ndarray:
    """
    Load embeddings from:
      - .npy  (shape inferred from header; supports mmap)
      - .npz  (shape inferred from stored array; loads into memory)
      - raw headerless memmap file (requires --dim; N inferred from file size)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    suffix = os.path.splitext(path)[1].lower()

    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings in {path}, got shape {arr.shape}")
        return arr

    if suffix == ".npz":
        with np.load(path) as z:
            if npz_key is None:
                if len(z.files) != 1:
                    raise ValueError(f"{path} contains multiple arrays {z.files}; specify --db-npz-key.")
                npz_key = z.files[0]
            arr = z[npz_key]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings in {path}, got shape {arr.shape}")
        return arr

    if dim is None:
        raise ValueError(
            "Raw memmap input has no header, so d cannot be inferred. "
            "Provide --db-dim, and N will be inferred from file size."
        )

    dt = np.dtype(dtype)
    d = int(dim)
    byte_size = os.path.getsize(path)
    denom = dt.itemsize * d
    if byte_size % denom != 0:
        raise ValueError(
            f"File size {byte_size} is not divisible by dtype_size({dt.itemsize}) * d({d}). "
            "Check --db-dtype/--db-dim."
        )
    N = byte_size // denom
    return np.memmap(path, dtype=dt, mode="r", shape=(N, d))


class NameIndex:
    def __init__(self, txt_path: str, idx_path: str, encoding: str = "utf-8") -> None:
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Names file not found: {txt_path}")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"Names index file not found: {idx_path}")

        self._fh = open(txt_path, "rb")
        self._idx = np.memmap(idx_path, dtype=np.int64, mode="r")
        self._encoding = encoding

        if self._idx.size < 2:
            raise ValueError(f"Invalid index file (size < 2): {idx_path}")

    def __len__(self) -> int:
        return max(int(self._idx.size) - 1, 0)

    def get(self, i: int) -> str:
        if i < 0 or i + 1 >= self._idx.size:
            return ""
        start = int(self._idx[i])
        end = int(self._idx[i + 1])
        if end <= start:
            return ""
        self._fh.seek(start)
        data = self._fh.read(end - start)
        if data.endswith(b"\n"):
            data = data[:-1]
        return data.decode(self._encoding, errors="replace")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def read_faiss_index(index_path: str, mmap: bool = False) -> faiss.Index:
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if mmap:
        return faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    return faiss.read_index(index_path)


def _set_nprobe_recursive(index: faiss.Index, nprobe: int) -> int:
    """
    Attempt to set nprobe on IVF-like indices, including wrapped indices.
    Returns the number of sub-indices updated.
    """
    updated = 0

    # Many FAISS indices expose nprobe directly (IndexIVF and some wrappers)
    if hasattr(index, "nprobe"):
        try:
            index.nprobe = int(nprobe)
            updated += 1
        except Exception:
            pass

    # Handle common wrappers
    # IndexPreTransform: underlying index is in .index
    if isinstance(index, faiss.IndexPreTransform):
        updated += _set_nprobe_recursive(index.index, nprobe)

    # IndexIDMap / IndexIDMap2: underlying index is in .index
    if isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2):
        updated += _set_nprobe_recursive(index.index, nprobe)

    # IndexShards: per-shard indices accessible via at(i)
    if isinstance(index, faiss.IndexShards):
        try:
            for si in range(index.count()):
                updated += _set_nprobe_recursive(index.at(si), nprobe)
        except Exception:
            pass

    # IndexReplicas: per-replica indices accessible via at(i)
    if isinstance(index, faiss.IndexReplicas):
        try:
            for ri in range(index.count()):
                updated += _set_nprobe_recursive(index.at(ri), nprobe)
        except Exception:
            pass

    return updated


def maybe_set_nprobe(index: faiss.Index, nprobe: Optional[int]) -> None:
    if nprobe is None:
        return
    updated = _set_nprobe_recursive(index, int(nprobe))
    if updated > 0:
        eprint(f"[faiss] nprobe set to {nprobe} on {updated} index object(s).")
    else:
        eprint("[faiss] nprobe not applied (index may not be IVF-based).")


def clamp_topk(k: int, index: faiss.Index) -> int:
    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal <= 0:
        raise ValueError("Index appears empty (ntotal <= 0).")
    if k <= 0:
        raise ValueError("top-k must be > 0.")
    return min(int(k), ntotal)


def search_in_batches(
    index: faiss.Index,
    queries: np.ndarray,
    topk: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search queries in batches and concatenate results.
    Returns (D, I) with shapes [N, K].
    """
    nq = queries.shape[0]
    K = topk

    all_D = np.empty((nq, K), dtype=np.float32)
    all_I = np.empty((nq, K), dtype=np.int64)

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        q_batch = queries[start:end]
        D, I = index.search(q_batch, K)

        # FAISS typically returns float32 distances; enforce consistent dtypes
        all_D[start:end] = np.asarray(D, dtype=np.float32)
        all_I[start:end] = np.asarray(I, dtype=np.int64)

        eprint(f"[search] processed queries {start}:{end} / {nq}")

    return all_D, all_I


def rerank_exact(
    queries: np.ndarray,
    cand_I: np.ndarray,
    db_emb: np.ndarray,
    topk: int,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rerank candidates using exact dot-product (or cosine if normalize=True).
    Returns (D, I) with shapes [N, topk].
    """
    nq = queries.shape[0]
    out_D = np.empty((nq, topk), dtype=np.float32)
    out_I = np.empty((nq, topk), dtype=np.int64)

    for i in range(nq):
        ids = cand_I[i]
        valid = ids >= 0
        ids = ids[valid]
        if ids.size == 0:
            out_I[i] = -1
            out_D[i] = -np.inf
            continue

        vecs = np.asarray(db_emb[ids], dtype=np.float32)
        if normalize:
            if not vecs.flags.writeable:
                vecs = vecs.copy()
            l2_normalize_inplace(vecs)

        scores = vecs @ queries[i]
        k = min(topk, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        out_I[i, :k] = ids[idx]
        out_D[i, :k] = scores[idx]
        if k < topk:
            out_I[i, k:] = -1
            out_D[i, k:] = -np.inf

    return out_D, out_I


def main():
    parser = argparse.ArgumentParser(description="Unified FAISS search (FlatIP / IVF / IVFPQ, etc.)")
    parser.add_argument("--query", required=True, help="Path to query embeddings (.npy or .pkl)")
    parser.add_argument("--index", required=True, help="Path to FAISS index file")
    parser.add_argument("--output", required=True, help="Output .npz path (will store indices and distances)")

    parser.add_argument("--topk", type=int, default=1000, help="Requested top-k (will be clamped to index.ntotal)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Query batch size. If 0, uses min(num_queries, 1000).",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="nprobe for IVF-based indices (ignored for Flat indices).",
    )
    parser.add_argument(
        "--mmap_index",
        action="store_true",
        help="Memory-map the FAISS index file (may reduce RAM usage).",
    )
    parser.add_argument(
        "--no_mmap_npy",
        action="store_true",
        help="Disable numpy mmap for .npy query loading.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed (smaller file, slower).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize queries (use with --normalize when building the index for cosine).",
    )
    parser.add_argument("--db-names", default=None, help="DB names text file (<prefix>.names.txt)")
    parser.add_argument("--db-names-idx", default=None, help="DB names index file (<prefix>.names.idx)")
    parser.add_argument("--query-names", default=None, help="Query names text file (<prefix>.names.txt)")
    parser.add_argument("--query-names-idx", default=None, help="Query names index file (<prefix>.names.idx)")
    parser.add_argument("--output-tsv", default=None, help="Optional TSV output with names")

    parser.add_argument("--db-embeddings", default=None, help="DB embeddings for rerank (.npy/.npz/raw)")
    parser.add_argument("--db-dim", type=int, default=128, help="Embedding dim for raw db embeddings")
    parser.add_argument("--db-dtype", type=str, default="float16", help="dtype for raw db embeddings")
    parser.add_argument("--db-npz-key", type=str, default=None, help="Array key for .npz db embeddings")
    parser.add_argument("--rerank", type=int, default=0, help="If >0, rerank using exact vectors with candidate k")

    args = parser.parse_args()

    eprint(f"[load] query: {args.query}")
    queries = load_queries(args.query, mmap_npy=not args.no_mmap_npy)
    nq, d = queries.shape
    eprint(f"[load] queries shape: {queries.shape}, dtype={queries.dtype}")
    if args.normalize:
        if not queries.flags.writeable:
            queries = queries.copy()
        l2_normalize_inplace(queries)
        eprint("[config] queries L2-normalized")

    eprint(f"[load] index: {args.index} (mmap={args.mmap_index})")
    index = read_faiss_index(args.index, mmap=args.mmap_index)
    eprint(f"[load] index ntotal={int(index.ntotal)}")

    # Optional: set nprobe for IVF-like indices
    maybe_set_nprobe(index, args.nprobe)

    # Clamp topk to index size
    topk = clamp_topk(args.topk, index)
    eprint(f"[config] topk requested={args.topk}, clamped={topk}")

    # Candidate size for optional rerank
    if args.rerank and args.rerank > 0:
        cand_k = max(int(args.rerank), topk)
        cand_k = clamp_topk(cand_k, index)
        eprint(f"[config] rerank enabled: candidate_k={cand_k}, final_topk={topk}")
    else:
        cand_k = topk

    # Auto batch size rule: min(num_queries, 1000)
    if args.batch_size and args.batch_size > 0:
        batch_size = int(args.batch_size)
    else:
        batch_size = min(nq, 1000)
    batch_size = max(1, batch_size)
    eprint(f"[config] batch_size={batch_size} (num_queries={nq})")

    # Dimension check when available
    if hasattr(index, "d"):
        index_d = int(index.d)
        if index_d != d:
            raise ValueError(f"Dimension mismatch: queries D={d} but index D={index_d}")

    eprint("[search] searching...")
    D, I = search_in_batches(index=index, queries=queries, topk=cand_k, batch_size=batch_size)

    if args.rerank and args.rerank > 0:
        if not args.db_embeddings:
            raise ValueError("--rerank requires --db-embeddings")
        eprint(f"[rerank] loading db embeddings: {args.db_embeddings}")
        db_emb = load_embeddings_from_path(
            args.db_embeddings,
            dtype=args.db_dtype,
            npz_key=args.db_npz_key,
            dim=args.db_dim,
        )
        if db_emb.shape[1] != d:
            raise ValueError(f"DB embeddings dim mismatch: db D={db_emb.shape[1]} vs query D={d}")
        eprint("[rerank] computing exact scores...")
        D, I = rerank_exact(queries=queries, cand_I=I, db_emb=db_emb, topk=topk, normalize=args.normalize)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    eprint(f"[save] writing: {args.output}")
    if args.compressed:
        np.savez_compressed(args.output, indices=I, distances=D)
    else:
        np.savez(args.output, indices=I, distances=D)

    if args.output_tsv:
        if (args.db_names is None) != (args.db_names_idx is None):
            raise ValueError("Provide both --db-names and --db-names-idx for name mapping.")
        if (args.query_names is None) != (args.query_names_idx is None):
            raise ValueError("Provide both --query-names and --query-names-idx for query name mapping.")

        db_names = NameIndex(args.db_names, args.db_names_idx) if args.db_names else None
        q_names = NameIndex(args.query_names, args.query_names_idx) if args.query_names else None

        if db_names is not None and len(db_names) < int(index.ntotal):
            eprint(f"[warn] db names count {len(db_names)} < index ntotal {int(index.ntotal)}")

        out_dir = os.path.dirname(os.path.abspath(args.output_tsv))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        eprint(f"[save] writing: {args.output_tsv}")
        with open(args.output_tsv, "w", encoding="utf-8") as f:
            f.write("query_idx\\tquery_name\\thit_rank\\thit_id\\thit_name\\tscore\\n")
            for qi in range(nq):
                qn = q_names.get(qi) if q_names else ""
                for rank in range(topk):
                    hid = int(I[qi, rank])
                    if hid < 0:
                        continue
                    hn = db_names.get(hid) if db_names else ""
                    f.write(f"{qi}\t{qn}\t{rank}\t{hid}\t{hn}\t{float(D[qi, rank]):.6f}\n")

        if db_names is not None:
            db_names.close()
        if q_names is not None:
            q_names.close()

    eprint("[done]")


if __name__ == "__main__":
    main()
