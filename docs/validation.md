# Validation of the 2026-09-11 update

These checks validate software behavior. They are not a rerun of the paper's
full training experiments, generation-quality evaluations, or large-scale benchmarks.
The published model architecture/checkpoint tensors were not edited.

## Environment

- Linux compute node with NVIDIA H800 80 GB GPUs
- Existing `ProTok` conda environment; Python 3.11, PyTorch 2.5.1,
  Lightning 2.5.1.post0
- Published `ProTok_main.ckpt` and existing GFP CSV files; no data/checkpoint downloads
- Isolated source/test directory; one- and two-GPU runs

## Automated regression tests

```bash
python -m unittest discover -s tests -v
```

13 tests passed. Coverage includes:

- actual AdamW updates with zero gradients: only ordinary weights decay;
- complete/disjoint optimizer groups, frozen parameter exclusion, and legacy mode;
- learning-rate warmup/decay endpoints and no rebound beyond the schedule;
- custom CSV columns, optional labels, training-only quantile edges, deterministic
  split membership/order, and explicit preprocessing policies;
- nonfinite targets, invalid sequences and cross-split leakage rejection;
- replica padding removal, original input ordering and global regression metrics;
- batched regression prediction, sampling helpers, and beam-search termination
  for inputs finishing at different steps.

For the published model plus regression head, the corrected optimizer groups were:

| Group | Parameter tensors | Scalar parameters | Weight decay |
| --- | ---: | ---: | ---: |
| Ordinary trainable weights | 154 | 123,959,344 | 0.0001 |
| Bias/norm/embedding parameters | 149 | 1,219,025 | 0 |

Frozen CLIP projection parameters are excluded from both groups.

## Full GFP data validation

```bash
python -m scripts.transfer_learning \
  --train_csv_path data/Generation_data/DMS/GFP/GFP-train.csv \
  --test_csv_path data/Generation_data/DMS/GFP/GFP-test.csv \
  --validate_data_only
```

The default seed-42 split yielded **32,143 training**, **5,673 validation**, and
**16,208 test** rows. Classes 1–8 mapped to 0–7, with no exact-sequence overlap
between splits. GFP sequences were 237 residues, without `J` padding.

## Real-checkpoint training, evaluation and export

Short fixtures used the first 9 rows of the GFP training CSV for training, the next
5 for validation, and the first 5 test rows for testing. The odd split sizes exercise
sampler padding and uneven prediction partitions on two GPUs. Each run used one
epoch, batch size 2 per GPU, gradient accumulation 2, `max_len=320`, and zero workers.

```bash
python -m scripts.transfer_learning \
  --train_csv_path results/smoke/train.csv \
  --val_csv_path results/smoke/val.csv \
  --test_csv_path results/smoke/test.csv \
  --num_gpus 2 --num_epoch 1 --batch_size 2 --num_workers 0 \
  --max_len 320 --accumulate_grad_batches 2 \
  --ckpt_path results/smoke/ddp_final_ckpt \
  --save_embedding_path results/smoke/ddp_final.pkl \
  --logger_save_dir results/smoke/logs --project_name ddp_final \
  --log_every_n_steps 1 --progress_refresh_rate 0
```

The corresponding single-GPU run used separate output paths and `--num_gpus 1`.
Both automatically selected BF16 mixed precision on H800.

| Check | One GPU | Two GPUs |
| --- | --- | --- |
| Actual AdamW updates in saved optimizer state | 3 | 2 |
| Training → test → export | Passed | Passed |
| Export shape | `(9, 64, 12)` | `(9, 64, 12)` |
| Float32 finite embeddings, int64 labels | Passed | Passed |
| Source row, target and class-label alignment | Passed | Passed |
| Reload through inference wrapper with `strict=True` | Passed | Passed |
| Beam search, top-p and greedy decoding, batch of 2 | Passed | Passed |

The first encoder query-projection weight changed by a maximum absolute amount of
approximately `2.00e-4` (one GPU) and `1.50e-4` (two GPUs), confirming training updated
the backbone. Different GPU counts change the effective batch and optimizer-step
count; these runs are not intended to produce identical trained weights.

An initial FP16 run completed its stages but skipped all three attempted optimizer
updates due to GradScaler overflow. This motivated the BF16 default on supported
GPUs. The final validation checks optimizer state and weight changes, not just
successful process completion. Explicit FP16 remains available.

## Encoding and inference

Five input FASTA records were encoded with the published checkpoint using one GPU
and `torchrun --nproc_per_node=2`, batch size 2. The resulting `(5, 768)` arrays
matched in original input order with **maximum absolute difference 0.0**. The
unconditional export had shape `(5, 64, 12)`.

Python syntax checks and `git diff --check` passed. Broader changes were limited to
verified defects and documentation. Diffusion training quality, full sequence
search throughput, and phylogenetic benchmark accuracy were not re-evaluated.
