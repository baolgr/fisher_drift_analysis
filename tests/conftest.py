import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.resnet_cifar import build_resnet50_cifar
from src.models.vit_cifar import build_vit_small_cifar


@pytest.fixture
def tiny_resnet():
    """Reduced-depth ResNet50 (1 block per stage) -- same classes as the full
    model, just fewer repetitions, for fast forward/backward/hook tests that
    don't depend on the full layer registry (see test_fisher_metric.py for
    registry-resolution tests, which use the full-depth model instead)."""
    return build_resnet50_cifar(layers=(1, 1, 1, 1))


@pytest.fixture
def tiny_vit():
    """Reduced-depth ViT (2 blocks) -- same reasoning as tiny_resnet."""
    return build_vit_small_cifar(depth=2)


@pytest.fixture
def dummy_cifar_batch():
    torch.manual_seed(0)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


def make_synthetic_dataloaders(n_train=128, n_val=32, n_test=32, batch_size=32):
    """CIFAR-sized random tensors, no real dataset/download involved --
    used by the slow end-to-end smoke tests to exercise the real train.py
    pipeline quickly."""
    from torch.utils.data import DataLoader, TensorDataset

    def _make(n):
        images = torch.randn(n, 3, 32, 32)
        labels = torch.randint(0, 10, (n,))
        return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True, drop_last=True)

    return _make(n_train), _make(n_val), _make(n_test)
