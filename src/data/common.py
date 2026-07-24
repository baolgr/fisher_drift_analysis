"""Shared helpers for the per-dataset loaders in this package -- data-root
resolution and the seeded train/val split used identically by every dataset
that only ships an official train+test split (no separate val).
"""

from pathlib import Path
from typing import List, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_root(data_root: str) -> str:
    path = Path(data_root)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


def seeded_train_val_split(n_total: int, n_val: int, seed: int) -> Tuple[List[int], List[int]]:
    """Reproducible (seeded) split of range(n_total) into (train, val) index
    lists, val sized n_val. Shared across cifar10/cifar100/imagenet32 so a
    given seed carves out the same relative split fraction consistently.
    """
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[:n_train], perm[n_train:]
