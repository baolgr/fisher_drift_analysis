"""CIFAR-100 data pipeline: dataloaders for the FisherAdapTune benchmark.

Mirrors cifar10.py exactly (same torchvision-backed pattern, same seeded
45k/5k-style split) -- only the dataset class and normalization constants
differ. torchvision.datasets.CIFAR100 namespaces itself under its own
subfolder of data_root, so it can share the same `dataset/` root as CIFAR-10
without collision.
"""

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.data.common import resolve_data_root, seeded_train_val_split

_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train_loader, val_loader, test_loader) for CIFAR-100.

    train_loader/val_loader come from a reproducible 45k/5k split of the
    official 50k train set (seeded by cfg["seed"]); test_loader is the
    official 10k test set, held out entirely from training/validation and
    meant to be evaluated exactly once, at the end of a run.
    """
    data_root = resolve_data_root(cfg["data_root"])
    seed = int(cfg.get("seed", 0))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ]
    )

    # download=False -- see cifar10.py for why (no-internet compute nodes).
    train_full = CIFAR100(root=data_root, train=True, download=False, transform=train_transform)
    val_full = CIFAR100(root=data_root, train=True, download=False, transform=eval_transform)
    test_set = CIFAR100(root=data_root, train=False, download=False, transform=eval_transform)

    n_val = int(cfg.get("val_size", 5000))
    train_indices, val_indices = seeded_train_val_split(len(train_full), n_val, seed)

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)

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
