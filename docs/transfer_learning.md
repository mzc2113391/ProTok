# Transfer learning with your own data

ProTok jointly learns a scalar regression target and reconstructs the input sequence.
Prepare CSV files and choose column names on the command line; no Python dataset class is needed.
Run all commands from the repository root after activating `ProTok`.

## CSV format

The minimum input has a protein sequence and a numeric target:

```csv
sequence,activity
ACDEFGHIK,0.12
ACDEFGHIL,0.75
```

This illustrates the schema only; training needs enough independent sequences for a
training split and at least two validation rows. Targets must be finite numbers.
Use the same target units/scaling across all splits. ProTok does not automatically
normalize targets: the relative scale affects the regression/reconstruction losses.

```bash
python -m scripts.transfer_learning \
  --train_csv_path my_data/train.csv \
  --val_csv_path my_data/validation.csv \
  --test_csv_path my_data/test.csv \
  --sequence_column sequence \
  --target_column activity \
  --init_ckpt checkpoint/ProTok_main.ckpt \
  --num_gpus 1 \
  --ckpt_path results/my_transfer/checkpoints \
  --save_embedding_path results/my_transfer/train_embeddings.pkl \
  --project_name my_transfer
```

- `--val_csv_path` is optional. Without it, 85% of unique sequences are selected for
  training with seed 42; `--val_fraction` and `--dataseed` change this split.
  Replicates of the same preprocessed sequence stay in the same split.
- `--test_csv_path` is optional and is evaluated only after checkpoint selection.
  Supply disjoint splits. Shared sequences after preprocessing cause an error.
  For homologous sequences, supply family/group-aware splits to assess generalization;
  the automatic split groups exact duplicates, not sequence similarity.
- `--num_gpus 0` uses CPU; `--num_gpus 2` uses two GPUs.
  `--batch_size` is per device. Gradient accumulation is controlled with
  `--accumulate_grad_batches`. The scheduler counts actual optimizer steps.
  Supported GPUs default to `bf16-mixed`; other GPUs use `16-mixed`, and CPU uses
  `32-true`. Override with `--precision`. BF16 avoids FP16 loss-scaler overflows
  observed in initial short runs with the published checkpoint on H800.
- `--validate_data_only` checks data without loading a checkpoint or starting training.
  It assumes 64 prefix tokens unless `--num_prefix` is given. During training, the
  prefix count is read from the checkpoint and any override must match it.
- `--max_len` includes prefix tokens and BOS/EOS. With the published 64-prefix
  checkpoint and default 1024, at most 958 amino acids fit. Longer inputs raise an
  error unless `--long_sequences truncate` explicitly enables cropping.
- Sequences are trimmed at their ends and converted to uppercase. The 20 standard
  amino acids and `X` are accepted. `--unknown_residues map-to-x` maps ambiguous
  letters to `X`; punctuation and internal whitespace remain errors.
  `--strip_characters J` explicitly reproduces the old GFP padding removal when needed.

## Optional diffusion labels

Regression needs only the target. A separate categorical column is used exclusively
for conditional diffusion export. By default, a `label` column is used if present;
choose another with `--label_column class`, or disable it with `--label_column ''`.
Validation/test files do not need class labels.

String labels, 0-based integers and 1-based integers are mapped to contiguous IDs
`0 .. num_classes - 1`, in sorted order of the values present in the training split.
The mapping is saved with the export. For the full GFP training set, labels 1–8 map
to 0–7. When a class is absent from a custom training split, inspect the saved mapping
before selecting diffusion classes.

If only a continuous target is available, `--num_bins 8` creates quantile classes
using **training targets only**, overriding the categorical column. Repeated quantile
edges reduce the number of classes and produce a warning; use the saved `num_classes`.
No categorical labels are required for transfer learning itself.

```bash
python -m scripts.transfer_learning \
  --train_csv_path my_data/train.csv \
  --sequence_column sequence \
  --target_column activity \
  --num_bins 8 \
  --num_gpus 1
```

## Outputs and model selection

The default checkpoint criterion is minimum validation MSE. Choose
`--monitor val_pearson` or `--monitor val_spearman` for correlation-based selection.
Correlations are undefined for constant targets/predictions; prefer MSE for those
cases. The best checkpoint and `last.ckpt` are saved; `--save_top_k` controls retention.
Use a fresh output directory for each experiment.

The TensorBoard run directory contains:

- `run_config.json`: command-line settings, including the optimizer mode;
- `data_manifest.json`: input paths, seed, zero-based source CSV row indices for each
  split, and the diffusion class mapping/bin edges;
- training, validation and optional test metrics.

After training, the best checkpoint is used for testing and for exporting **training
rows only**. Validation/test rows are excluded from the diffusion training pickle.
The pickle contains:

| Key | Meaning |
| --- | --- |
| `embedding` | Float32 `(N, num_prefix, latent_dim)` array; normally `(N, 64, 12)` |
| `labels` | Optional int64 class IDs `(N,)` |
| `row_indices` | Zero-based row indices in the original training CSV |
| `targets` | Regression targets, aligned with embeddings |
| `metadata` | Exact split manifest and class mapping/bin edges |

Embedding order follows the saved training split order; `row_indices` maps it back
to the source CSV. Single- and multi-GPU export use the same ordering. Distributed
validation/test metrics remove sampler padding before computing global statistics.
Reconstruction loss is aggregated with the same square-root sequence-length weights
used by the training objective.

Use `--skip_export` for regression-only experiments. To inspect export metadata:

```python
import pickle
with open('results/transfer_runs/train_embeddings.pkl', 'rb') as handle:
    data = pickle.load(handle)
print(data['embedding'].shape)
print(data['metadata']['labels'])
```

For conditional diffusion, pass this pickle to `dit_lightning.train --train_pkl`
and set `--num_classes` to the saved count. For a pickle without labels, pass
`--no_labels`. See the repository README for the rest of the diffusion workflow.

## GFP compatibility and reproducibility

```bash
python -m scripts.transfer_learning \
  --train_csv_path data/Generation_data/DMS/GFP/GFP-train.csv \
  --test_csv_path data/Generation_data/DMS/GFP/GFP-test.csv \
  --num_gpus 2 \
  --monitor val_pearson \
  --save_embedding_path results/GFP/train_embeddings.pkl \
  --ckpt_path results/GFP/checkpoints \
  --project_name GFP
```

For unique sequences, the default 85:15 split with seed 42 preserves the original
sklearn split membership and order. Default loss weights remain reconstruction 1,
regression 2; change them explicitly with `--recon_loss_weight` and `--reg_loss_weight`.

The previous optimizer groups were inverted: ordinary weights received no decay,
while bias/norm/embedding parameters received the configured decay. This release
corrects that assignment. `--legacy_weight_decay` restores the inverted assignment
for comparisons, but does **not** restore all old behavior: the scheduler, evaluation
aggregation and checkpoint selection also have fixes. For exact historical code,
use the original commit `b202958`. Existing checkpoints remain loadable; the model
architecture and published weights are unchanged. New training results can differ,
so record the Git revision and do not substitute short smoke-test metrics for paper
benchmark results.

## Checks

```bash
python -m unittest discover -s tests -v
python -m scripts.transfer_learning --help
```

The regression tests cover actual AdamW decay behavior, scheduler endpoints,
CSV validation and split reproducibility, train-only label fitting, distributed
ordering/padding, batched regression prediction, and decoding helper functions.
