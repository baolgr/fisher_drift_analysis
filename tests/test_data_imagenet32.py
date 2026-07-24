"""Tests for src/data/imagenet32.py -- the one genuinely new dataset parser
added alongside CIFAR-100 (cifar10.py/cifar100.py are thin torchvision
wrappers, untested here same as the pre-existing precedent for cifar10.py).
Byte layout (channel-planar reshape) and label re-indexing (1-indexed on
disk -> 0-indexed for CrossEntropyLoss) are the parts most likely to have an
off-by-one bug, so this builds tiny synthetic pickle batches (no download)
and exercises build_dataloaders end to end.
"""

import pickle

import numpy as np
import pytest

from src.data.imagenet32 import build_dataloaders

_NUM_CLASSES = 5


def _make_batch(path, n, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(n, 3072), dtype=np.uint8)
    labels = rng.integers(1, _NUM_CLASSES + 1, size=n).tolist()  # 1-indexed on disk
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)


@pytest.fixture
def staged_data_root(tmp_path):
    _make_batch(tmp_path / "train_data_batch_1", n=20, seed=0)
    _make_batch(tmp_path / "train_data_batch_2", n=20, seed=1)
    _make_batch(tmp_path / "val_data", n=15, seed=2)
    return tmp_path


def _cfg(data_root, val_size=10):
    return {
        "data_root": str(data_root),
        "seed": 0,
        "batch_size": 8,
        "num_workers": 0,
        "val_size": val_size,
    }


def test_split_sizes(staged_data_root):
    train_loader, val_loader, test_loader = build_dataloaders(_cfg(staged_data_root, val_size=10))
    assert len(val_loader.dataset) == 10
    assert len(train_loader.dataset) == 40 - 10  # pooled train_data_batch_1 + _2
    assert len(test_loader.dataset) == 15  # val_data, used as the held-out test set


def test_image_shape_and_label_range(staged_data_root):
    train_loader, _, _ = build_dataloaders(_cfg(staged_data_root))
    images, labels = next(iter(train_loader))
    assert images.shape[1:] == (3, 32, 32)
    assert labels.min() >= 0
    assert labels.max() < _NUM_CLASSES  # 1-indexed on disk -> 0-indexed here


def test_missing_val_size_raises(staged_data_root):
    cfg = _cfg(staged_data_root)
    del cfg["val_size"]
    with pytest.raises(ValueError):
        build_dataloaders(cfg)


def test_missing_train_batches_raises_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataloaders(_cfg(tmp_path))
