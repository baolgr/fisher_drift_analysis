"""Tests specific to ResNetCIFAR / build_resnet50_cifar."""

import torch
import torch.nn as nn
import pytest

from src.fisher.adafisher import AdaFisherBackbone
from src.models.resnet_cifar import build_resnet50_cifar

from conftest import make_synthetic_dataloaders


def test_output_shape(dummy_cifar_batch):
    images, _ = dummy_cifar_batch
    model = build_resnet50_cifar()
    logits = model(images)
    assert logits.shape == (images.shape[0], 10)


def test_param_count_sane():
    model = build_resnet50_cifar()
    n_params = sum(p.numel() for p in model.parameters())
    assert 20_000_000 <= n_params <= 25_000_000, f"unexpected param count: {n_params}"


def test_no_inplace_relu():
    model = build_resnet50_cifar()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, "found an inplace ReLU -- will crash AdaFisher hooks"


def test_backward_survives_hooks(tiny_resnet, dummy_cifar_batch):
    images, labels = dummy_cifar_batch
    AdaFisherBackbone(tiny_resnet, TCov=1)
    logits = tiny_resnet(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()  # must not raise "modified by an inplace operation"


@pytest.mark.slow
@pytest.mark.parametrize("disable_freeze", [True, False])
def test_train_smoke(monkeypatch, disable_freeze, tmp_path):
    import src.train as train_mod

    monkeypatch.setattr(train_mod, "build_dataloaders", lambda cfg: make_synthetic_dataloaders())

    cfg = {
        "data_root": "unused",
        "seed": 0,
        "batch_size": 32,
        "num_workers": 0,
        "num_epochs": 2,
        "learning_rate": 3e-4,
        "weight_decay": 5e-5,
        "early_stopping_patience": 10,
        "output_dir": str(tmp_path),
        "fisher_ema_interval": 2,
        "freeze_interval": 4,
        "fisher_slice_mode": "row",
        "fisher_slice_blocks": 2,
        "js_distance_mode": "log",
        "chunk_selection_metric": "total_variation",
        "js_variance_lambda": 1.0,
    }
    summary = train_mod.run_training("resnet50", cfg, disable_freeze=disable_freeze)

    assert "test_accuracy" in summary
    assert "trainable_params" in summary and "total_params" in summary

    if disable_freeze:
        ratio = summary["trainable_params"] / summary["total_params"]
        assert ratio > 0.99, f"unexpected freezing with disable_freeze=True: ratio={ratio}"
    else:
        assert summary["trainable_params"] < summary["total_params"]

    run_dir = tmp_path / f"resnet50_{'nofreeze' if disable_freeze else 'freeze'}"
    plots_dir = run_dir / "metrics_plots"
    for name in ("fisher_drift", "fisher_magnitude", "grad_norm", "relative_update", "all_metrics_grid"):
        png = plots_dir / f"{name}.png"
        assert png.exists() and png.stat().st_size > 0
    assert (run_dir / "summary.json").exists()
