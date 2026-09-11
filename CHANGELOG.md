# Release notes

## 2026-09-11 — Transfer learning and inference correctness

### Fixed

- Correct inverted AdamW weight-decay groups in transfer learning. Bias,
  normalization and embedding parameters are exempt; other trainable weights
  receive the configured decay. Frozen CLIP parameters remain excluded.
- Calculate warmup/cosine scheduling from Lightning's actual optimizer-step count,
  including distributed sampling and gradient accumulation. Clamp the cosine
  schedule at its minimum instead of allowing it to rise after the final step.
- Remove unnecessary differentiable prediction/target gathers from the MSE
  training objective. DDP handles gradient averaging.
- Compute validation/test regression metrics after removing duplicate sampler
  padding; aggregate reconstruction loss by its sequence-length weights.
- Keep training, test and embedding export in the same distributed strategy.
  Restore sample order before exporting training embeddings and labels.
- Restore FASTA input order in multi-GPU `scripts.encode`; the old rank-wise
  concatenation could associate embeddings with the wrong sequence names.
- Fix decoding completion state to be per input for beam search, provide the two
  missing sampling helper functions, validate decoding arguments, and support
  batched regression predictions.
- Handle output filenames without a directory in encoding and tree inference.
- Correct README environment naming, download/extraction guidance, shell line
  continuations, and the command that creates unconditional diffusion input.

### Added

- A generic CSV DataModule with configurable sequence, scalar target and optional
  class columns, separate validation/test files, and reproducible automatic splits.
- Explicit sequence handling options, finite-target validation, and protection
  against exact-sequence leakage across splits.
- Optional target quantile bins fitted only on training data, contiguous class ID
  mapping, and export metadata linking embeddings to source CSV rows.
- CPU support, data-only validation, configurable loss weights/checkpoint monitor,
  bounded checkpoint retention, and a documented custom-data workflow.
- Automatic BF16 mixed precision on supported GPUs, avoiding the initial FP16
  gradient-scaler overflows observed on H800; an explicit precision override remains available.
- Automated regression tests; see [validation notes](docs/validation.md) for the
  environment and scope of cluster smoke tests.

### Reproducibility

The model architecture and published checkpoint tensors are unchanged. The original
release used inverted weight decay and different scheduling/evaluation behavior;
newly trained models may produce different results. `--legacy_weight_decay` is for
isolating the old decay assignment, not full historical reproduction. Use commit
`b202958` for the original code, and record a commit ID with benchmark runs.

The default checkpoint monitor is now validation MSE; use `--monitor val_pearson`
to retain the GFP example's selection criterion. At most the best checkpoint plus
`last.ckpt` are retained by default. Validation/test data are never included in the
conditional training-embedding export.
