"""CIFAR-10 data pipeline: download + train/val/test dataloaders.

Single entry point (build_dataloaders) so there is no separate download
script to run first, and no import-order dependency between a "download"
step and a "build dataloaders" step.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _resolve_data_root(data_root: str) -> str:
    path = Path(data_root)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train_loader, val_loader, test_loader) for CIFAR-10.

    train_loader/val_loader come from a reproducible 45k/5k split of the
    official 50k train set (seeded by cfg["seed"]); test_loader is the
    official 10k test set, held out entirely from training/validation and
    meant to be evaluated exactly once, at the end of a run.
    """
    data_root = _resolve_data_root(cfg["data_root"])
    seed = int(cfg.get("seed", 0))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )

    # download=False: fail fast with a clear error if `dataset/` wasn't staged, rather than
    # attempting a network fetch -- compute nodes on HPC clusters (e.g. the Alliance Canada
    # H100 nodes this benchmark also runs on) typically have no internet access, so a stale
    # or missing dataset/ would otherwise hang/fail with an opaque network timeout instead.
    train_full = CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
    val_full = CIFAR10(root=data_root, train=True, download=False, transform=eval_transform)
    test_set = CIFAR10(root=data_root, train=False, download=False, transform=eval_transform)

    n_total = len(train_full)
    n_val = 5000
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices, val_indices = perm[:n_train], perm[n_train:]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

    # pin_memory speeds up host->device transfer on CUDA specifically; a no-op on
    # CPU/MPS, so this is safe to set unconditionally rather than threading a new
    # config key through for it.
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader
