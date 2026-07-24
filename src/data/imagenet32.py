"""Downsampled ImageNet (32x32) data pipeline: dataloaders for the
FisherAdapTune benchmark.

Expected layout under cfg["data_root"] (the official Chrabaszcz et al.
"Downsampled ImageNet" distribution, 32x32 variant -- NOT ImageFolder, NOT
downloaded by this code, see CLAUDE.md's "no dataset downloads" rule):

    <data_root>/train_data_batch_1
    <data_root>/train_data_batch_2
    ...
    <data_root>/train_data_batch_N   (however many batches were staged)
    <data_root>/val_data

Each file is a Python-2-era pickle (hence encoding="latin1" below) holding a
dict with keys 'data' ((N, 3072) uint8, channel-planar -- reshape(-1,3,32,32)
gives CHW directly, no transpose needed) and 'labels' (1-indexed class ids,
1..1000). `val_data` is the official held-out set and is used here as the
**test** set (mirrors cifar10.py/cifar100.py's "official test set touched
exactly once" convention) -- the train/val split used during training is
instead carved out of the pooled train_data_batch_* files, seeded like the
CIFAR loaders.

The full pooled train array is loaded into memory (~4GB for the 1000-class
variant) -- same in-memory-Dataset pattern already used for CIFAR via
torchvision, just done by hand here since there's no torchvision class for
this format. A known future perf concern (memory-mapping, lazy per-batch
loading) if it becomes a problem in practice, not solved here.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.common import resolve_data_root, seeded_train_val_split

# Downsampled ImageNet has no widely-used normalization constants of its own;
# standard ImageNet mean/std are the common choice in the literature for this variant.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ImageNet32(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx].transpose(1, 2, 0))  # CHW -> HWC for PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def _load_pickle_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        # encoding="latin1": these pickles were produced under Python 2 by the
        # official distribution; plain pickle.load fails to unpickle them under 3.
        batch = pickle.load(f, encoding="latin1")
    return np.asarray(batch["data"]), np.asarray(batch["labels"])


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build (train_loader, val_loader, test_loader) for Downsampled ImageNet (32x32).

    Unlike cifar10.py/cifar100.py, there is no fixed default val split size --
    the train pool here is ~25x larger than CIFAR's, so a config must set
    "val_size" explicitly rather than inheriting a CIFAR-sized default.
    """
    data_root = Path(resolve_data_root(cfg["data_root"]))
    seed = int(cfg.get("seed", 0))
    batch_size = int(cfg.get("batch_size", 128))
    num_workers = int(cfg.get("num_workers", 2))
    val_size = cfg.get("val_size")
    if val_size is None:
        raise ValueError(
            "imagenet32 has no CIFAR-sized default val split (the train pool is ~25x "
            "larger) -- set 'val_size' explicitly in the config."
        )
    val_size = int(val_size)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )

    train_batch_paths = sorted(data_root.glob("train_data_batch_*"))
    if not train_batch_paths:
        raise FileNotFoundError(
            f"No train_data_batch_* files found under {data_root} -- stage the Downsampled "
            "ImageNet 32x32 train batches there before training (see this module's docstring "
            "for the expected layout)."
        )
    val_path = data_root / "val_data"
    if not val_path.exists():
        raise FileNotFoundError(
            f"{val_path} not found -- stage the Downsampled ImageNet val batch there "
            "(see this module's docstring for the expected layout)."
        )

    images_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for path in train_batch_paths:
        images, labels = _load_pickle_batch(path)
        images_list.append(images)
        labels_list.append(labels)
    train_images = np.concatenate(images_list, axis=0).reshape(-1, 3, 32, 32).astype(np.uint8)
    train_labels = np.concatenate(labels_list, axis=0).astype(np.int64) - 1  # 1-indexed -> 0-indexed

    test_images, test_labels = _load_pickle_batch(val_path)
    test_images = test_images.reshape(-1, 3, 32, 32).astype(np.uint8)
    test_labels = test_labels.astype(np.int64) - 1

    train_indices, val_indices = seeded_train_val_split(len(train_labels), val_size, seed)

    train_set = _ImageNet32(train_images[train_indices], train_labels[train_indices], train_transform)
    val_set = _ImageNet32(train_images[val_indices], train_labels[val_indices], eval_transform)
    test_set = _ImageNet32(test_images, test_labels, eval_transform)

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
