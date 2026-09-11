"""Order-preserving assembly of distributed predictions."""
import numpy as np


def merge_indexed_arrays(parts, expected_size=None):
    """Restore input order and remove DistributedSampler padding by sample index."""
    nonempty = [part for part in parts if len(part['index'])]
    if not nonempty:
        raise ValueError('No predictions were produced.')
    merged = {key: np.concatenate([part[key] for part in nonempty]) for key in nonempty[0]}
    indices, first = np.unique(merged['index'], return_index=True)
    if expected_size is not None and not np.array_equal(indices, np.arange(expected_size)):
        raise RuntimeError(f'Incomplete predictions: expected {expected_size} rows, got {len(indices)}.')
    return {key: value[first] for key, value in merged.items()}
