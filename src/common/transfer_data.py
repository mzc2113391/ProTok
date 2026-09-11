"""CSV input for supervised transfer learning; no dataset subclass is required."""

from pathlib import Path
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .utils import feature_generate, restype_order
from .label_weights import DistributedWeightedSampler, sample_label_weights


class ProteinCSVDataset(Dataset):
    """Validated sequences and scalar targets, with source CSV row indices."""

    def __init__(self, frame, num_prefix_tokens=64, max_len=1024, regression_weights=None):
        self.frame = frame.reset_index(drop=True)
        self.num_prefix = num_prefix_tokens
        self.max_len = max_len
        self.data_seqs = self.frame["sequence"].tolist()
        self.data_targets = torch.tensor(self.frame["target"].to_numpy(), dtype=torch.float32)
        self.row_indices = self.frame["row_index"].to_numpy(dtype=np.int64)
        self.regression_weights = regression_weights

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        sequence = self.data_seqs[index]
        features = feature_generate(
            {
                "seq_len": len(sequence),
                "aatype": np.array([restype_order.get(aa, restype_order["X"]) for aa in sequence]),
                "residue_index": np.arange(1, len(sequence) + 1),
            },
            num_prefix_tokens=self.num_prefix,
            max_len=self.max_len,
            mask=False,
        )
        item = {"input": features, "fitness": self.data_targets[index], "index": index}
        if self.regression_weights is not None:
            item["regression_weight"] = np.float32(self.regression_weights[index])
        return item


class ProteinDataModule(L.LightningDataModule):
    """Read CSV columns, split by unique sequence, and fit export labels on training rows."""

    def __init__(
        self, train_csv_path, test_csv_path=None, batch_size=32, num_prefix=64,
        max_len=1024, num_workers=4, val_csv_path=None, sequence_column="seq",
        target_column="fitness", label_column="label", val_fraction=0.15,
        seed=42, num_bins=None, unknown_residues="error", long_sequences="error",
        strip_characters="", sampler_label_weights=(1.0,), reg_loss_label_weights=(1.0,),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds = self.val_ds = self.test_ds = self.predict_ds = None
        self.export_labels = None
        self.label_metadata = {}
        self.sampling_weights = None
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative.")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1.")
        if max_len <= num_prefix + 2:
            raise ValueError("max_len must be > num_prefix + 2 (including special tokens).")
        if num_bins is not None and num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

    def _read(self, path):
        h = self.hparams
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        raw = pd.read_csv(path)
        required = {h.sequence_column, h.target_column}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}; available: {list(raw.columns)}")
        if raw.empty:
            raise ValueError(f"{path}: CSV contains no rows.")
        sequences = []
        canonical = set("ARNDCQEGHILKMFPSTWYVX")
        for row, value in enumerate(raw[h.sequence_column], start=2):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: empty or invalid sequence at CSV line {row}.")
            sequence = value.strip().upper()
            if h.strip_characters:
                sequence = sequence.translate(str.maketrans("", "", h.strip_characters.upper()))
            if not sequence:
                raise ValueError(f"{path}: sequence at CSV line {row} is empty after stripping.")
            unknown = set(sequence) - canonical
            if unknown:
                if h.unknown_residues == "error" or not sequence.isalpha() or not sequence.isascii():
                    raise ValueError(f"{path}: unsupported residues {sorted(unknown)} at CSV line {row}; "
                                     "use --unknown_residues map-to-x for ambiguous amino acids.")
                sequence = "".join(aa if aa in canonical else "X" for aa in sequence)
            limit = h.max_len - h.num_prefix - 2
            if len(sequence) > limit:
                if h.long_sequences == "error":
                    raise ValueError(f"{path}: sequence at CSV line {row} has {len(sequence)} residues; "
                                     f"maximum for the configured max_len is {limit}. Increase --max_len to retain the full sequence.")
                sequence = sequence[:limit]
            sequences.append(sequence)
        targets = pd.to_numeric(raw[h.target_column], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(targets).all():
            rows = (np.flatnonzero(~np.isfinite(targets)) + 2).tolist()
            raise ValueError(f"{path}: nonnumeric or nonfinite targets at CSV lines {rows[:10]}.")
        frame = pd.DataFrame({"sequence": sequences, "target": targets, "row_index": np.arange(len(raw))})
        if h.label_column and h.label_column in raw:
            frame["class_label"] = raw[h.label_column]
        return frame

    def _dataset(self, frame):
        return ProteinCSVDataset(frame, self.hparams.num_prefix, self.hparams.max_len)

    @staticmethod
    def _check_overlap(left, right, names):
        overlap = set(left["sequence"]) & set(right["sequence"])
        if overlap:
            raise ValueError(f"{names}: {len(overlap)} shared sequences after preprocessing; "
                             "provide disjoint splits to avoid data leakage.")

    def _fit_labels(self, train):
        h = self.hparams
        if h.num_bins is not None:
            edges = np.unique(np.quantile(train["target"], np.linspace(0, 1, h.num_bins + 1)))
            if len(edges) < 3:
                raise ValueError("Training targets cannot form two distinct quantile bins; supply class labels.")
            self.export_labels = np.searchsorted(edges[1:-1], train["target"], side="right").astype(np.int64)
            self.label_metadata = {"method": "training_quantiles", "edges": edges.tolist(),
                                   "num_classes": len(edges) - 1}
            if len(edges) - 1 != h.num_bins:
                warnings.warn(f"Tied target values reduced num_classes to {len(edges) - 1}.")
        elif "class_label" in train:
            if train["class_label"].isna().any():
                raise ValueError("Training class labels contain missing values.")
            labels, values = pd.factorize(train["class_label"], sort=True)
            self.export_labels = labels.astype(np.int64)
            self.label_metadata = {"method": "column", "column": h.label_column,
                                   "classes": values.tolist(), "num_classes": len(values)}
        else:
            self.export_labels = None
            self.label_metadata = {"method": "none", "num_classes": 0}

    def setup(self, stage=None):
        if self.train_ds is None:
            full = self._read(self.hparams.train_csv_path)
            if self.hparams.val_csv_path:
                train, val = full, self._read(self.hparams.val_csv_path)
            else:
                unique = full["sequence"].drop_duplicates().to_numpy()
                train_size = int(len(unique) * (1 - self.hparams.val_fraction))
                if train_size < 1 or len(unique) - train_size < 2:
                    raise ValueError("Split needs at least one training and two validation sequences; "
                                     "provide --val_csv_path or increase --val_fraction/data size.")
                train_seqs, val_seqs = train_test_split(unique, train_size=train_size, random_state=self.hparams.seed)
                # Preserve the legacy sklearn split order for CSVs with unique sequences.
                if len(unique) == len(full):
                    indexed = full.set_index("sequence", drop=False)
                    train = indexed.loc[train_seqs].reset_index(drop=True)
                    val = indexed.loc[val_seqs].reset_index(drop=True)
                else:
                    grouped = full.groupby("sequence", sort=False)
                    train = pd.concat([grouped.get_group(seq) for seq in train_seqs], ignore_index=True)
                    val = pd.concat([grouped.get_group(seq) for seq in val_seqs], ignore_index=True)
            if len(val) < 2:
                raise ValueError("Validation needs at least two rows for correlation metrics.")
            self._check_overlap(train, val, "Training/validation")
            self._fit_labels(train)
            self.train_ds, self.val_ds = self._dataset(train), self._dataset(val)
            self.sampling_weights = sample_label_weights(
                self.hparams.sampler_label_weights, self.export_labels,
                self.label_metadata["num_classes"], "sampler_label_weights",
            )
            self.train_ds.regression_weights = sample_label_weights(
                self.hparams.reg_loss_label_weights, self.export_labels,
                self.label_metadata["num_classes"], "reg_loss_label_weights",
            )
            self.predict_ds = self.train_ds
            if self.hparams.test_csv_path:
                test = self._read(self.hparams.test_csv_path)
                self._check_overlap(train, test, "Training/test")
                self._check_overlap(val, test, "Validation/test")
                self.test_ds = self._dataset(test)

    def split_manifest(self):
        """Zero-based source row indices make the exact split reproducible."""
        self.setup()
        return {
            "seed": self.hparams.seed,
            "train_csv": str(self.hparams.train_csv_path),
            "val_csv": str(self.hparams.val_csv_path or self.hparams.train_csv_path),
            "test_csv": str(self.hparams.test_csv_path) if self.hparams.test_csv_path else None,
            "train_rows": self.train_ds.row_indices.tolist(),
            "val_rows": self.val_ds.row_indices.tolist(),
            "test_rows": self.test_ds.row_indices.tolist() if self.test_ds is not None else [],
            "labels": self.label_metadata,
        }

    def _loader(self, dataset, shuffle=False, sampler=None):
        if dataset is None:
            raise ValueError("Requested split is not configured.")
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, sampler=sampler,
                          num_workers=self.hparams.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        if self.sampling_weights is not None:
            trainer = self.trainer
            sampler = DistributedWeightedSampler(
                self.train_ds, self.sampling_weights,
                num_replicas=trainer.world_size if trainer is not None else 1,
                rank=trainer.global_rank if trainer is not None else 0, seed=self.hparams.seed,
            )
            return self._loader(self.train_ds, sampler=sampler)
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)

    def predict_dataloader(self):
        return self._loader(self.predict_ds)
