import os
import pickle as pkl
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def _pad_1d(x: np.ndarray, length: int, value=0):
    if x.shape[0] >= length:
        return x[:length]
    pad = np.full((length - x.shape[0],), value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _pad_2d(x: np.ndarray, length: int, value=0):
    if x.shape[0] >= length:
        return x[:length]
    pad = np.full((length - x.shape[0],) + x.shape[1:], value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


class EmbeddingDataset(Dataset):
    """Simple dataset for epsilon-pred training.
    Expects a pickle with keys:
      - embedding: (N, T, F) float array in [-1,1]
      - labels: (N,) int (optional)
    """

    def __init__(self, pkl_path: str, seq_len: int, use_labels: bool = True):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            data = pkl.load(f)
        self.emb = data['embedding']  # can be array or list
        if use_labels and 'labels' in data:
            self.labels = np.asarray(data['labels'])
        else:
            self.labels = None
        self.seq_len = seq_len

    def __len__(self):
        return self.emb.shape[0] if hasattr(self.emb, 'shape') else len(self.emb)

    def __getitem__(self, idx):
        x = self.emb[idx]  # (T,F) possibly list or array
        x = np.asarray(x)
        T = x.shape[0]
        if T <= self.seq_len:
            start = 0
        else:
            start = np.random.randint(0, T - self.seq_len + 1)
        end = min(start + self.seq_len, T)
        x = x[start:end]
        x = _pad_2d(x, self.seq_len, value=0)
        seq_mask = np.zeros((self.seq_len,), dtype=np.bool_)
        seq_mask[:(end - start)] = True
        residue_index = np.arange(self.seq_len, dtype=np.int16)
        label = int(self.labels[idx]) if self.labels is not None else 0

        return {
            'embedding': torch.from_numpy(x).float(),
            'labels': torch.tensor(label, dtype=torch.long),
            'seq_mask': torch.from_numpy(seq_mask),
            'residue_index': torch.from_numpy(residue_index),
        }


def _load_single_file(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pkl.load(f)


class ProtupleDataset(Dataset):
    """Dataset for x0-pred training.
    Input list contains file paths to pickles with a dict that at least has `base` entry.
    We select `base` variant and crop/pad to `seq_len` as JAX code does.
    Output keys: protokens, aatypes, seq_mask, residue_index
    """

    def __init__(self, list_path: str, seq_len: int):
        super().__init__()
        with open(list_path, 'r') as f:
            self.paths = [line.strip() for line in f if line.strip()]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.paths)

    def _get_crop_idx(self, feature: Dict[str, np.ndarray]) -> (int, int):
        L = int(feature['seq_mask'].shape[-1])
        if L <= self.seq_len:
            return 0, min(L, self.seq_len)
        seq_len_true = int(feature['seq_mask'].sum())
        if seq_len_true <= self.seq_len:
            return 0, self.seq_len
        start = np.random.randint(0, seq_len_true - self.seq_len + 1)
        return start, start + self.seq_len

    def _crop(self, feat: Dict[str, np.ndarray], start: int, end: int):
        new = {}
        new['seq_mask'] = feat['seq_mask'][start:end]
        new['aatype'] = feat['aatype'][start:end]
        new['code_indices'] = feat['code_indices'][start:end]
        new['residue_index'] = feat['residue_index'][start:end]
        return new

    def __getitem__(self, idx):
        d = _load_single_file(self.paths[idx])
        base = d['base'] if 'base' in d else d  # fallback if already flat
        start, end = self._get_crop_idx(base)
        crop = self._crop(base, start, end)
        # pad to seq_len
        aatypes = _pad_1d(crop['aatype'].astype(np.int16), self.seq_len, value=0)
        protokens = _pad_1d(crop['code_indices'].astype(np.int16), self.seq_len, value=0)
        seq_mask = _pad_1d(crop['seq_mask'].astype(np.bool_), self.seq_len, value=0)
        residue_index = _pad_1d(crop['residue_index'].astype(np.int16), self.seq_len, value=0)
        return {
            'aatypes': torch.from_numpy(aatypes),
            'protokens': torch.from_numpy(protokens),
            'seq_mask': torch.from_numpy(seq_mask),
            'residue_index': torch.from_numpy(residue_index),
        }


@dataclass
class DataConfig:
    train_pkl: Optional[str] = None
    train_list: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    seq_len: int = 256
    use_labels: bool = True


class DiTDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig, x_prediction: bool = False):
        super().__init__()
        self.cfg = cfg
        self.x_prediction = x_prediction

    def setup(self, stage: Optional[str] = None):
        if self.x_prediction:
            assert self.cfg.train_list, 'train_list is required for x_prediction mode'
            self.train_ds = ProtupleDataset(self.cfg.train_list, self.cfg.seq_len)
        else:
            assert self.cfg.train_pkl, 'train_pkl is required for epsilon-pred mode'
            self.train_ds = EmbeddingDataset(self.cfg.train_pkl, self.cfg.seq_len, use_labels=self.cfg.use_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
