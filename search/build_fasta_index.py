#!/usr/bin/env python3
"""
Build a random-access name/sequence index from a FASTA file.

Outputs (prefix = --output-prefix):
  - <prefix>.names.txt   (one name per line)
  - <prefix>.names.idx   (int64 byte offsets, length N+1)
  - <prefix>.seqs.txt    (one sequence per line, optional)
  - <prefix>.seqs.idx    (int64 byte offsets, length N+1, optional)
  - <prefix>.meta.json   (basic metadata)

The .idx format allows O(1) random access to any name/sequence without
loading the entire list into memory.
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Iterator, Optional, Tuple


def iter_fasta(path: Path, *, read_seqs: bool) -> Iterator[Tuple[str, Optional[str]]]:
    name = None
    seq_chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    if read_seqs:
                        yield name, "".join(seq_chunks)
                    else:
                        yield name, None
                name = line[1:]
                if read_seqs:
                    seq_chunks = []
            else:
                if read_seqs:
                    seq_chunks.append(line)

        if name is not None:
            if read_seqs:
                yield name, "".join(seq_chunks)
            else:
                yield name, None


def write_index(txt_path: Path, idx_path: Path, items: Iterator[str]) -> int:
    offsets = array("Q")
    offsets.append(0)
    offset = 0
    count = 0

    with txt_path.open("wb") as f:
        for item in items:
            data = item.encode("utf-8") + b"\n"
            f.write(data)
            offset += len(data)
            offsets.append(offset)
            count += 1

    # Write offsets as int64
    with idx_path.open("wb") as f:
        offsets.tofile(f)

    return count


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build random-access name/sequence index from FASTA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input FASTA path")
    p.add_argument("--output-prefix", required=True, help="Output prefix path")
    p.add_argument("--write-seqs", action="store_true", help="Also write sequences index")
    args = p.parse_args()

    fasta_path = Path(args.input)
    if not fasta_path.exists():
        raise FileNotFoundError(str(fasta_path))

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    names_txt = Path(str(out_prefix) + ".names.txt")
    names_idx = Path(str(out_prefix) + ".names.idx")
    seqs_txt = Path(str(out_prefix) + ".seqs.txt")
    seqs_idx = Path(str(out_prefix) + ".seqs.idx")
    meta_path = Path(str(out_prefix) + ".meta.json")

    # Stream FASTA once, optionally writing sequences
    if args.write_seqs:
        name_offsets = array("Q")
        seq_offsets = array("Q")
        name_offsets.append(0)
        seq_offsets.append(0)
        name_off = 0
        seq_off = 0
        count = 0

        with names_txt.open("wb") as nf, seqs_txt.open("wb") as sf:
            for name, seq in iter_fasta(fasta_path, read_seqs=True):
                name_data = name.encode("utf-8") + b"\n"
                seq_data = (seq or "").encode("utf-8") + b"\n"
                nf.write(name_data)
                sf.write(seq_data)
                name_off += len(name_data)
                seq_off += len(seq_data)
                name_offsets.append(name_off)
                seq_offsets.append(seq_off)
                count += 1

        with names_idx.open("wb") as f:
            name_offsets.tofile(f)
        with seqs_idx.open("wb") as f:
            seq_offsets.tofile(f)
    else:
        def name_iter_only():
            for name, _ in iter_fasta(fasta_path, read_seqs=False):
                yield name

        count = write_index(names_txt, names_idx, name_iter_only())

    meta = {
        "input_fasta": str(fasta_path),
        "count": int(count),
        "names": {"txt": str(names_txt), "idx": str(names_idx)},
    }
    if args.write_seqs:
        meta["seqs"] = {"txt": str(seqs_txt), "idx": str(seqs_idx)}

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Indexed FASTA: {count} records")
    print(f"Names: {names_txt} / {names_idx}")
    if args.write_seqs:
        print(f"Seqs: {seqs_txt} / {seqs_idx}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
