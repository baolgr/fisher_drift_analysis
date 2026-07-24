"""Dataset registry + dispatcher -- the single stable entrypoint train.py and
scripts/regenerate_plots.py import from.

Adding a future dataset: write src/data/<name>.py with a build_dataloaders(cfg)
matching the (train_loader, val_loader, test_loader) signature used by
cifar10.py/cifar100.py/imagenet32.py, then add one DatasetSpec entry to
DATASET_REGISTRY below. Nothing else needs to change -- src/train.py reads
num_classes/display_name from this registry rather than hardcoding them.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from torch.utils.data import DataLoader

from src.data import cifar10, cifar100, imagenet32


@dataclass(frozen=True)
class DatasetSpec:
    build_dataloaders: Callable[[Dict], Tuple[DataLoader, DataLoader, DataLoader]]
    num_classes: int  # default classifier head size; a config can override via num_classes:
    display_name: str  # runs/<display_name>/... folder name


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(cifar10.build_dataloaders, num_classes=10, display_name="CIFAR-10"),
    "cifar100": DatasetSpec(cifar100.build_dataloaders, num_classes=100, display_name="CIFAR-100"),
    "imagenet32": DatasetSpec(imagenet32.build_dataloaders, num_classes=1000, display_name="ImageNet"),
}


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    key = cfg.get("dataset", "cifar10")
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{key}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key].build_dataloaders(cfg)
