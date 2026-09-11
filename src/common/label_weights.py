"""Class weights for sampling and scalar regression."""

import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler


def parse_label_weights(value):
    """Accept one shared weight or comma-separated weights in class-ID order."""
    try:
        weights = tuple(float(item.strip()) for item in value.split(','))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('Use comma-separated numeric label weights, e.g. 1,1,1.') from exc
    if not weights or not np.isfinite(weights).all() or min(weights) < 0 or max(weights) == 0:
        raise argparse.ArgumentTypeError('Label weights must be finite, nonnegative, and not all zero.')
    return weights


def sample_label_weights(weights, labels, num_classes, option):
    """Expand class weights onto training rows; normalize to avoid numeric overflow."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any() or values.max() == 0:
        raise ValueError(f'{option}: weights must be finite, nonnegative, and not all zero.')
    if len(values) == 1:
        # A single weight applies equally to every row, with or without labels.
        return None
    if labels is None:
        raise ValueError(f'{option}: multiple class weights require --label_column or --num_bins.')
    if len(values) != num_classes:
        raise ValueError(f'{option}: expected {num_classes} weights in class-ID order, got {len(values)}.')
    per_row = (values / values.max())[labels]
    if not per_row.any():
        raise ValueError(f'{option}: at least one observed training class must have positive weight.')
    return None if np.all(per_row == per_row[0]) else per_row


class DistributedWeightedSampler(DistributedSampler):
    """Draw a shared weighted stream, then shard it equally across training ranks."""

    def __init__(self, dataset, weights, num_replicas=1, rank=0, seed=42):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if len(self.weights) != len(dataset):
            raise ValueError('Sampling weights must contain one entry per training row.')

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank:self.total_size:self.num_replicas].tolist())


def weighted_mse_loss(prediction, target, weights=None):
    """Weighted mean squared error, with the global denominator for DDP gradients."""
    errors = F.mse_loss(prediction.float().flatten(), target.float().flatten(), reduction='none')
    if weights is None:
        return errors.mean()
    weights = weights.to(device=errors.device, dtype=torch.float32).flatten()
    numerator = (errors * weights).sum()
    denominator = weights.sum().detach()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    # DDP averages parameter gradients. This scale recovers the global weighted mean.
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return numerator * world_size / denominator
