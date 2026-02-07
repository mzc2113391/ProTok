#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ProTok phylogenetic congruence screening

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import lightning as L
from tqdm import tqdm

from net.ProTok import ProTok_lightning  
import pandas as pd
from src.common.utils import create_feature_list, read_fasta
from src.common.phylo import (
    cal_correlation,
    parse_fasta_to_dict,
    run_incongruence_pipeline,
    rf_distance,
    create_tree,
)


def setup_logger(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )


def banner(title: str) -> None:
    line = "═" * max(12, len(title) + 6)
    logging.info("\n%s\n  %s\n%s", line, title, line)


def kv(key: str, value: str, key_width: int = 32) -> None:
    logging.info(f"{key:<{key_width}} : {value}")


def resolve_device(device: str) -> torch.device:
    device = device.strip().lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_1d_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().reshape(-1).cpu().numpy()


def corr_stat(x) -> float:
    """Robustly convert correlation outputs to float.
    Supports: float, numpy scalar, (stat, p) tuple, scipy.stats.SignificanceResult.
    """
    if x is None:
        return float("nan")
    # SciPy >= 1.10: SignificanceResult(statistic=..., pvalue=...)
    if hasattr(x, "statistic"):
        return float(x.statistic)
    # Old style: (statistic, pvalue)
    if isinstance(x, (tuple, list)) and len(x) > 0:
        return float(x[0])
    # Numpy scalar / float-like
    return float(x)


# -----------------------------
# Core pipeline
# -----------------------------
@dataclass
class RunOutputs:
    spearman_rho: float
    norm_rf: float
    is_hgt_detected: bool
    is_high_detected: bool
    df_path: Optional[Path] = None
    tree_query_path: Optional[Path] = None
    tree_ref_path: Optional[Path] = None


@torch.no_grad()
def embed_query_sequences(
    model: ProTok_lightning,
    names: List[str],
    seqs: List[str],
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Encode each query sequence into a single embedding vector (1D numpy).
    """
    data_dict_list = create_feature_list(seqs, device=model.device)
    iterator = enumerate(data_dict_list)
    if show_progress:
        iterator = tqdm(iterator, total=len(data_dict_list), desc="Embedding query sequences")

    out: Dict[str, np.ndarray] = {}
    for idx, data_dict in iterator:
        output, _clip = model.encode(data_dict)
        out[names[idx]] = to_1d_numpy(output)
    return out


@torch.no_grad()
def embed_reference_multiseq(
    model: ProTok_lightning,
    refseq_dict: Dict[str, List[str]],
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Encode each reference 'taxon/group' that may contain multiple sequences.
    We embed each sequence and then flatten-concatenate into one long vector.
    """
    iterator = refseq_dict.items()
    if show_progress:
        iterator = tqdm(list(iterator), desc="Embedding reference groups")

    out: Dict[str, np.ndarray] = {}
    for phylo_name, sequences in iterator:
        input_list = create_feature_list(sequences, device=model.device)
        seq_embs: List[np.ndarray] = []
        for i in range(len(sequences)):
            output, _clip = model.encode(input_list[i])
            seq_embs.append(to_1d_numpy(output))
        out[phylo_name] = np.concatenate(seq_embs, axis=0)  # shape: (k*D,)
    return out


def compute_metrics_and_trees(
    phylo_embedding_dict: Dict[str, np.ndarray],
    ref_embedding_dict: Dict[str, np.ndarray],
    names_in_order: List[str],
    return_newick: bool = True,
) -> Tuple[float, str, str, float]:
    """
    Returns:
      rho, query_tree_newick, ref_tree_newick, normRF
    """
    seq_embedding_list, _phylum_list, seq_embedding_ref, _r, rho = cal_correlation(
        phylo_embedding_dict, ref_embedding_dict
    )
    rho_val = corr_stat(rho)
    _r_val = corr_stat(_r)  

    query_tree = create_tree(seq_embedding_list, names_in_order, return_str=return_newick)
    ref_tree = create_tree(seq_embedding_ref, names_in_order, return_str=return_newick)

    norm_rf = float(rf_distance(query_tree, ref_tree)["norm_rf"])
    return float(rho_val), str(query_tree), str(ref_tree), norm_rf


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_df_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def load_model(ckpt_path: Path, device: torch.device) -> ProTok_lightning:
    model = ProTok_lightning.load_from_checkpoint(str(ckpt_path), strict=False)
    model.to(device)
    model.eval()
    return model


def decide_screening(rho: float, rho_incongruent_threshold: float, rho_congruent_threshold: float) -> str:
    """
    Decision rules:
      - rho >= rho_congruent_threshold : "congruent" (skip HGT pipeline)
      - rho_incongruent_threshold < rho < rho_congruent_threshold : "no_significant" (skip congruence pipeline)
      - rho <= rho_incongruent_threshold : "run_congruence_pipeline"
    """
    if not np.isfinite(rho):
        return "run_congruence_pipeline"
    if rho >= rho_congruent_threshold:
        return "congruent"
    if (rho > rho_incongruent_threshold) and (rho < rho_congruent_threshold):
        return "no_significant"
    return "run_congruence_pipeline"


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ProTok phylogenetic congruence screening and HGT pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default="./checkpoint/ProTok_evo.ckpt", help="Path to .ckpt file")
    p.add_argument("--ref-fasta", type=str, default="./example/16S.fasta", help="Reference multi-seq FASTA")
    p.add_argument("--query-fasta", type=str, default="./example/glxK.fasta", help="Query FASTA (one seq per taxon)")
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0, ...")
    p.add_argument("--out-dir", type=str, default=None, help="If set, save outputs (trees + csv) to this folder")
    p.add_argument("--organism_info", type=str, default="./example/cog-24.org.csv", help="Organism information (taxa)")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")

    # NEW: screening thresholds
    p.add_argument(
        "--rho-incongruent-threshold",
        type=float,
        default=0.4,
        help="If rho <= this threshold, run run_congruence_pipeline (potentially incongruent).",
    )
    p.add_argument(
        "--rho-congruent-threshold",
        type=float,
        default=0.7,
        help="If rho >= this threshold, report as congruent and skip run_congruence_pipeline.",
    )
    return p


def main() -> RunOutputs:
    args = build_argparser().parse_args()
    setup_logger(args.verbose)

    ckpt_path = Path(args.checkpoint)
    ref_fasta = Path(args.ref_fasta)
    query_fasta = Path(args.query_fasta)
    out_dir = Path(args.out_dir) if args.out_dir else None
    show_progress = not args.no_progress

    device = resolve_device(args.device)

    # Validate thresholds
    if args.rho_incongruent_threshold >= args.rho_congruent_threshold:
        raise ValueError(
            f"Invalid thresholds: rho-hgt-threshold ({args.rho_incongruent_threshold}) must be < "
            f"rho-congruent-threshold ({args.rho_congruent_threshold})."
        )

    banner("Inputs")
    kv("Checkpoint", str(ckpt_path))
    kv("Reference FASTA", str(ref_fasta))
    kv("Query FASTA", str(query_fasta))
    kv("Device", str(device))
    kv("rho_incongruent_threshold", f"{args.rho_incongruent_threshold:.6f}")
    kv("rho_congruent_threshold", f"{args.rho_congruent_threshold:.6f}")

    banner("Load model")
    model = load_model(ckpt_path, device)
    kv("Model class", model.__class__.__name__)
    kv(
        "Model device",
        str(next(model.parameters()).device) if any(True for _ in model.parameters()) else str(model.device),
    )

    banner("Read FASTA")
    refseq_dict = parse_fasta_to_dict(str(ref_fasta))
    names, seqs = read_fasta(str(query_fasta))
    kv("#Reference groups", str(len(refseq_dict)))
    kv("#Query sequences", str(len(seqs)))

    banner("Embedding")
    ref_embedding_dict = embed_reference_multiseq(model, refseq_dict, show_progress=show_progress)
    phylo_embedding_dict = embed_query_sequences(model, names, seqs, show_progress=show_progress)

    banner("Correlation + Trees")
    rho, tree_query, tree_ref, norm_rf = compute_metrics_and_trees(
        phylo_embedding_dict, ref_embedding_dict, names_in_order=names, return_newick=True
    )
    kv("Spearman rho", f"{rho:.6f}")
    kv("normRF", f"{norm_rf:.6f}")

    # NEW: screening decision logic
    banner("Screening decision")
    decision = decide_screening(rho, args.rho_incongruent_threshold, args.rho_congruent_threshold)

    df = None
    is_hgt_detected = False
    is_high_detected = False

    if decision == "congruent":
        kv("Result", "Phylogenetically congruent: gene tree is consistent with the species tree.")
    elif decision == "no_significant":
        kv("Result", "no significant results")
    else:
        kv("Result", "rho <= threshold (run run_congruence_pipeline)")
        banner("HGT pipeline")
        org_df = pd.read_csv(args.organism_info, header=None, names=["GCF", "genome_name", "taxid", "phylum"])
        df, is_hgt_detected, is_high_detected = run_incongruence_pipeline(
            org_df, ref_embedding_dict, phylo_embedding_dict
        )
        kv("HGT detected", str(bool(is_hgt_detected)))

    # Save outputs if requested
    df_path = tree_q_path = tree_r_path = None
    if out_dir is not None:
        banner("Saving outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save trees (from compute_metrics_and_trees)
        tree_q_path = out_dir / "query_tree.newick"
        tree_r_path = out_dir / "ref_tree.newick"
        save_text(tree_q_path, tree_query + "\n")
        save_text(tree_r_path, tree_ref + "\n")
        kv("Saved query tree", str(tree_q_path))
        kv("Saved ref tree", str(tree_r_path))

        # Save pipeline table only if we actually ran it
        if df is not None:
            df_path = out_dir / "hgt_pipeline.csv"
            try:
                save_df_csv(df, df_path)
                kv("Saved pipeline table", str(df_path))
            except Exception as e:
                kv("Saved pipeline table", f"FAILED ({e})")
        else:
            kv("Saved pipeline table", "SKIPPED (screening decided no pipeline run)")

    banner("Done")
    return RunOutputs(
        spearman_rho=rho,
        norm_rf=norm_rf,
        is_hgt_detected=bool(is_hgt_detected),
        is_high_detected=bool(is_high_detected),
        df_path=df_path,
        tree_query_path=tree_q_path,
        tree_ref_path=tree_r_path,
    )


if __name__ == "__main__":
    main()