# ProTok

This is the official code release for paper **“Compressed protein manifolds for evolutionary reasoning and protein engineering.”**

<p align="center">
  <img src="https://github.com/mzc2113391/ProTok/blob/main/figs/git_cover.png" width="100%">
</p>

---

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Sequence encoding](#sequence-encoding)
- [Large-scale sequence search (FAISS)](#large-scale-sequence-search-faiss)
- [Tree inference & phylogenetic congruence](#tree-inference--phylogenetic-congruence)
- [Sequence design (latent diffusion + decoding)](#sequence-design-latent-diffusion--decoding)
- [Notes](#notes)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

**ProTok** compresses protein sequences into a compact latent representation that supports:

- **Fast embedding-based similarity search** (including large-scale FAISS retrieval)
- **Alignment-free evolutionary analyses** (e.g., tree inference and congruence screening)
- **Generative design** in latent space (diffusion sampling + sequence decoding)

---

## Installation

### Create a conda environment

This repository is configured via `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ProTok
```

### Core dependencies (reference)

Exact versions may vary across platforms; see `environment.yml` for the authoritative specification.

- Python 3.11
- PyTorch
- Lightning
- ete3
- scikit-bio
- Biopython
- FAISS (for large-scale search)

### Download data and checkpoints

Download the archives from the [ProTok data and checkpoint record](https://zenodo.org/records/18480835)
and extract the model files into `./checkpoint` and datasets into `./data`.
Use `tar -xJf` for `.tar.xz` archives and `unzip` for `.zip` archives, matching the
actual downloaded filenames. The published conda environment is a Linux/CUDA export.

> **Note:** The repo assumes the default checkpoint directory is `./checkpoint`. If you store files elsewhere, pass the corresponding paths in CLI arguments (see each module’s --help).

---

## Pretrained checkpoints

We provide three main pretrained checkpoints:

- **`ProTok_main.ckpt`**  
  Latent is **normalized** (recommended for **generation / design** workflows).

- **`ProTok_evo.ckpt`**  
  Latent has **no extra normalization** (recommended for **evolutionary reasoning** tasks).

- **`ProTok_search.ckpt`**  
  CLIP embedding for **sequence search** workflows (**retrieval** with CLIP embeddings + FAISS).

---

## Sequence encoding

### Single-GPU / CPU

```bash
python -m scripts.encode \
  --input ./example/your_sequence.fasta \
  --output ./results/embeddings.npy
```

### Multi-GPU encoding

```bash
torchrun --nproc_per_node=2 -m scripts.encode \
  --input ./example/your_sequence.fasta \
  --output ./results/embeddings.npy \
  --batch_size 128
```

> Tip: Use `python -m scripts.encode --help` to see all options (e.g., embedding type, batching).

---

## Large-scale sequence search (FAISS)

For retrieval, we recommend using **low-dimensional CLIP embeddings** and building a FAISS index.

### 1) Encode database and query FASTA files (CLIP)

```bash
python -m scripts.encode \
  --checkpoint ./checkpoint/ProTok_search.ckpt \
  --input ./example/exp_db.fasta \
  --output ./results/exp_db_embeddings.npy \
  --type clip

python -m scripts.encode \
  --checkpoint ./checkpoint/ProTok_search.ckpt \
  --input ./example/exp_qr.fasta \
  --output ./results/exp_qr_embeddings.npy \
  --type clip
```

### 2) Build FASTA name indices

```bash
python -m search.build_fasta_index \
  --input ./example/exp_db.fasta \
  --output-prefix ./results/exp_db

python -m search.build_fasta_index \
  --input ./example/exp_qr.fasta \
  --output-prefix ./results/exp_qr
```

### 3) (Optional) Pack `.npy` into a raw memmap (recommended for large databases)

```bash
python -m search.pack_embeddings \
  --input ./results/exp_db_embeddings.npy \
  --output ./results/exp_db_embeddings.fbin \
  --dtype float16
```

### 4) Build a FAISS index

**FlatIP** (exact; recommended for medium databases):

```bash
python -m search.build_faiss_index \
  --index-type flatip \
  --embeddings ./results/exp_db_embeddings.fbin \
  --output ./results/faiss_flatip.index
```

**IVFPQ** (approximate; recommended for 10M–100M+ scale):

```bash
python -m search.build_faiss_index \
  --index-type ivfpq \
  --embeddings ./results/exp_db_embeddings.fbin \
  --dim 128 \
  --dtype float16 \
  --output ./results/faiss_ivfpq.index
```

### 5) Run search

FlatIP:

```bash
python -m search.search \
  --query ./results/exp_qr_embeddings.npy \
  --index ./results/faiss_flatip.index \
  --topk 1000 \
  --db-names ./results/exp_db.names.txt \
  --db-names-idx ./results/exp_db.names.idx \
  --query-names ./results/exp_qr.names.txt \
  --query-names-idx ./results/exp_qr.names.idx \
  --output ./results/out_flat.npz \
  --output-tsv ./results/out_flat.tsv
```

IVFPQ with re-ranking:

```bash
python -m search.search \
  --query ./results/exp_qr_embeddings.npy \
  --index ./results/faiss_ivfpq.index \
  --topk 1000 \
  --nprobe 1024 \
  --rerank 5000 \
  --db-embeddings ./results/exp_db_embeddings.fbin \
  --output ./results/out_ivfpq.npz \
  --output-tsv ./results/out_ivfpq.tsv
```

---

## Tree inference & phylogenetic congruence

### Tree inference (alignment-free)

```bash
python -m scripts.tree_inference \
  --input ./example/TB2:S15435Taxa1_processed.fasta \
  --output_png ./results/my_tree.png
```

### Congruence analysis (screen gene–species discordance)

Prepare:
- **Reference marker FASTA** (example: `./example/16S.fasta`)
- **Query gene-family FASTA** (example: `./example/glxK.fasta`)
- **Organism information** (example: `./example/cog-24.org.csv`)
- Reference marker sequences should be named using:

```
<species_name>@<marker_name>
```

Run:

```bash
python -m scripts.congruence_analysis \
  --query-fasta ./example/glxK.fasta \
  --ref-fasta ./example/16S.fasta
```

---

## Sequence design (latent diffusion + decoding)

ProTok supports sequence design by:
1) encoding sequences into a latent training set,
2) training a latent diffusion model,
3) sampling latent embeddings,
4) decoding sampled embeddings back to sequences.

### Unconditional design (example workflow)

**Step 1 — prepare embeddings / training pickle**

```bash
python -m scripts.encode \
  --input ./example/your_sequence.fasta \
  --output ./results/train_embeddings.npy \
  --save_uncond_train_pkl ./example/uncond_traindit_exp.pkl
```

**Step 2 — train diffusion**

```bash
python -m dit_lightning.train \
  --train_pkl ./example/uncond_traindit_exp.pkl \
  --no_labels \
  --max_epochs 25000 \
  --save_dir ./results/dit_runs/uncond \
  --devices 2 \
  --tb_log_dir ./results/dit_runs/tensorboard_logger \
  --project_name uncond \
  --n_layers 12 \
  --hidden_size 512
```

**Step 3 — sample latent embeddings (DDP)**

```bash
torchrun --nproc_per_node=2 -m dit_lightning.infer_ddp \
  --ckpt_path ./checkpoint/uncond/luciferase_diff.ckpt \
  --samples_per_class 100 \
  --samples_per_device 50 \
  --labels 0 \
  --uncond \
  --out_pkl ./results/out_uncond.pkl
```

**Step 4 — decode to sequences**

```bash
# The decoder determines sequence length from the sampled embeddings. In practice, you can set --min-len and --max-len to the minimum/maximum lengths in the training set.
torchrun --nproc_per_node=2 -m scripts.seq_gen \
  --checkpoint ./checkpoint/ProTok_main.ckpt \
  --input-pkl ./results/out_uncond.pkl \
  --output-pkl ./results/out_uncond_seq.pkl \
  --batch-size 2 \
  --method beam_search \
  --num-beams 5 \
  --max-len 600 \
  --min-len 150
```

### Conditional design (GFP example)

**Step 1 — transfer learning**

Fine-tune ProTok with a CSV containing protein sequences, numeric fitness values,
and optional class labels. The GFP files use `seq`, `fitness`, and `label` columns:

```bash
SAMPLER_LABEL_WEIGHTS="1,1,1,1,1,1,1,1"
REG_LOSS_LABEL_WEIGHTS="1,1,1,1,1,1,1,1"

python -m scripts.transfer_learning \
  --num_gpus 2 \
  --train_csv_path ./data/Generation_data/DMS/GFP/GFP-train.csv \
  --test_csv_path ./data/Generation_data/DMS/GFP/GFP-test.csv \
  --sampler_label_weights "${SAMPLER_LABEL_WEIGHTS}" \
  --reg_loss_label_weights "${REG_LOSS_LABEL_WEIGHTS}" \
  --monitor val_pearson \
  --ckpt_path ./results/GFP/checkpoints \
  --save_embedding_path ./results/GFP/train_embeddings.pkl \
  --project_name GFP
```

Both weight settings default to **1 for every class** and can be omitted.
Values follow the sorted training-label order (GFP labels 1–8):

- `SAMPLER_LABEL_WEIGHTS` controls each class's per-example sampling weight.
  Nonuniform weights enable weighted sampling with replacement; all ones retain
  ordinary shuffled training.
- `REG_LOSS_LABEL_WEIGHTS` controls each class's contribution to the training
  regression loss, computed as `sum(weight * squared_error) / sum(weight)`.
  Reconstruction loss and validation/test metrics are unchanged.

For your own data, specify the column names:

```bash
python -m scripts.transfer_learning \
  --train_csv_path my_data/train.csv \
  --val_csv_path my_data/validation.csv \
  --sequence_column sequence \
  --target_column activity \
  --label_column class \
  --num_gpus 1
```

The class column is optional. Use `--num_bins 8` to create classes from training
fitness values when needed for conditional diffusion or class weighting.
Without `--val_csv_path`, 15% of the training CSV is reserved for validation;
`--val_fraction` and `--dataseed` control the split. `--test_csv_path` is optional.
Use `--validate_data_only` to check the CSV before training.
See the [data and training options](docs/transfer_learning.md) for details.

**Step 2 — train diffusion**

The export contains the training rows and a saved class mapping. Set `--num_classes`
to its recorded count (8 for the full GFP example).

```bash
python -m dit_lightning.train \
  --train_pkl ./results/GFP/train_embeddings.pkl \
  --max_epochs 25000 \
  --save_dir ./results/dit_runs/GFP \
  --devices 2 \
  --tb_log_dir ./results/dit_runs/tensorboard_logger \
  --project_name GFP \
  --n_layers 8 \
  --hidden_size 256 \
  --num_classes 8
```

**Step 3 — sample latent embeddings**

```bash
torchrun --nproc_per_node=2 -m dit_lightning.infer_ddp \
  --ckpt_path ./checkpoint/DMS/GFP_diff.ckpt \
  --samples_per_class 100 \
  --samples_per_device 100 \
  --labels 6,7 \
  --cfg_weight 1 \
  --out_pkl ./results/out_cond.pkl
```

**Step 4 — decode to sequences (fixed length for GFP)**

The example below uses the published transfer checkpoint. To use your newly trained
model, replace `--checkpoint` with the best checkpoint path printed in Step 1.
Likewise, replace diffusion checkpoint paths in Step 3 when using a newly trained model.

```bash
torchrun --nproc_per_node=2 -m scripts.seq_gen \
  --checkpoint ./checkpoint/DMS/GFP_transfer.ckpt \
  --input-pkl ./results/out_cond.pkl \
  --output-pkl ./results/out_cond_seq.pkl \
  --batch-size 2 \
  --method beam_search \
  --num-beams 5 \
  --max-len 237 \
  --min-len 237 \
  --cond
```

---

## Notes

- Use **`environment.yml`** for the published Linux/CUDA environment.
- For multi-GPU execution, make sure:
  - the CUDA driver/toolkit is correctly installed,
  - `torchrun` is available (PyTorch distributed runtime),
  - NCCL is properly configured and functional on your system.
- For dataset definitions and evaluation protocols used in our evolutionary reasoning benchmarks and sequence design experiments, please refer to:
  - **Evolutionary distance benchmark:** https://github.com/santule/pLMEvo  
  - **Phylogenetic tree reconstruction & taxonomic classification benchmark:** https://github.com/mims-harvard/Phyla  
  - **Sequence design data settings:** https://github.com/AzusaXuan/PRO-LDM

---

## Citation

If you use this codebase in your work, please cite:

```bibtex
@article{
}
```

---

## Contact

For questions, bug reports, or collaborations, please contact:

- **Zicheng Ma** — <mazicheng@stu.pku.edu.cn>
