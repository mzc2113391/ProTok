# Transfer learning options

See [Conditional design in the README](../README.md#conditional-design-gfp-example)
for the transfer-learning command and the subsequent diffusion workflow.

## Data

CSV files need a sequence column (`seq` by default) and a finite numeric target
(`fitness` by default). Choose column names with `--sequence_column` and
`--target_column`. Keep target units/scaling consistent across splits; targets are
not automatically normalized.

- `--val_csv_path`: optional independent validation CSV. Otherwise `--val_fraction`
  (default 0.15) and `--dataseed` (default 42) define the split. Duplicate sequences
  stay together; supplied train/validation/test splits must not share sequences.
- `--test_csv_path`: optional held-out test CSV, evaluated after model selection.
- `--label_column`: optional categorical column, default `label`. String and numeric
  values are mapped to IDs `0 .. num_classes - 1` in sorted training-label order.
  Use `--label_column ''` to disable it. Validation/test files need no class labels.
- `--num_bins`: creates target quantile classes using training rows only, overriding
  the class column. Tied quantile edges can reduce the number of classes.
- `--validate_data_only`: checks the data without loading weights or training.

Sequences are trimmed and uppercased. Standard amino acids and `X` are accepted;
`--unknown_residues map-to-x` maps ambiguous letters to `X`. Internal whitespace and
punctuation are rejected. `--strip_characters` explicitly removes specified characters.
`--max_len` includes prefix and special tokens: the default 1024 allows 958 amino
acids with the published checkpoint. Use `--long_sequences truncate` to permit cropping.

## Class weights

`--sampler_label_weights` and `--reg_loss_label_weights` accept comma-separated
weights in the sorted training-label order. For GFP labels 1–8:

```bash
--sampler_label_weights 1,1,1,1,1,1,1,1 \
--reg_loss_label_weights 1,1,1,1,1,1,1,1
```

Both default to a single `1`, meaning equal weights for every class. An explicit
list must contain one entry per class. Weights must be finite and nonnegative,
with at least one observed training class having positive weight.

Sampling weights multiply the sampling probability of each training example;
nonuniform weights enable sampling with replacement. An epoch has approximately
as many draws as training rows, with equal rank lengths for multi-GPU training.
Uniform weights retain ordinary shuffled training. These weights do not perform
automatic inverse-frequency class balancing.

Regression weights affect the training MSE through
`sum(weight * squared_error) / sum(weight)`. They do not change reconstruction loss,
validation/test metrics, or the complete training-row embedding export. The two
settings are independent and can be combined. Nonuniform class weights require
a class column or `--num_bins`.

## Training and outputs

- `--num_gpus 0` uses CPU; positive values set the GPU count. `--batch_size` is per
  device; `--accumulate_grad_batches` controls gradient accumulation.
- `--precision` overrides automatic BF16 on supported GPUs, FP16 on other GPUs,
  or float32 on CPU.
- `--recon_loss_weight` and `--reg_loss_weight` set overall reconstruction and
  regression coefficients (defaults 1 and 2).
- `--monitor` selects `val_mse` (default), `val_pearson`, or `val_spearman` for the
  best checkpoint. Correlation requires nonconstant targets/predictions.
- `--ckpt_path` selects the checkpoint directory. `--save_top_k` controls how many
  best checkpoints to keep, alongside `last.ckpt`. Use a fresh directory per run.
- `--save_embedding_path` selects the training-embedding pickle; `--skip_export`
  disables this export.

The best checkpoint is used for testing and export. The pickle contains float32
`embedding` of shape `(N, 64, 12)` for the published checkpoint, optional int64
`labels`, source `row_indices`, regression `targets`, and class/split `metadata`.
Only training rows are exported, once each, regardless of sampling weights.

For conditional diffusion, pass this pickle to `dit_lightning.train --train_pkl`
and use the exported class count for `--num_classes`. For an export without labels,
use `--no_labels`. Run `python -m scripts.transfer_learning --help` for all options.
