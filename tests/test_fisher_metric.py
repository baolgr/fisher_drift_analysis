"""Tests for the Fisher metric pipeline itself: hook-based Fisher collection,
chunk/layer resolution, JS-distance math, freeze masking, and the
LayerMetricRecorder aggregation -- all independent of which architecture is
used (parametrized over tiny_resnet / tiny_vit where relevant), so
correctness of the metric is verified once, not duplicated per model.
"""

import numpy as np
import pytest
import torch

from src.fisher.adafisher import AdaFisherBackbone
from src.fisher.fisher_core import (
    apply_chunk_masks,
    build_all_chunk_specs,
    collect_fisher_tensors,
    index_param_to_module,
    js_distance_log,
    register_mask_hooks,
)
from src.models.resnet_cifar import build_resnet50_cifar
from src.models.vit_cifar import build_vit_small_cifar
from src.utils.layers import RESNET50_CIFAR_LAYERS, VIT_SMALL_LAYERS, resolve_layer_chunk_keys
from src.utils.metrics import LayerMetricRecorder


def _forward_backward_with_hooks(model, images, labels):
    opt = AdaFisherBackbone(model, TCov=1)
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    return opt


# ---------------------------------------------------------------------------
# Fisher tensor collection
# ---------------------------------------------------------------------------


def test_fisher_nonzero_on_attention(tiny_vit, dummy_cifar_batch):
    images, labels = dummy_cifar_batch
    opt = _forward_backward_with_hooks(tiny_vit, images, labels)
    ftensors = collect_fisher_tensors(tiny_vit, opt)
    fisher = ftensors["blocks.0.attn.qkv.weight"]
    assert fisher.norm().item() > 0, (
        "Fisher on attention qkv is exactly zero -- this is exactly what would happen "
        "if nn.MultiheadAttention were used instead of a custom Linear-based Attention."
    )


def test_fisher_nonzero_on_conv_and_linear(tiny_resnet, tiny_vit, dummy_cifar_batch):
    images, labels = dummy_cifar_batch

    opt_r = _forward_backward_with_hooks(tiny_resnet, images, labels)
    ft_r = collect_fisher_tensors(tiny_resnet, opt_r)
    assert ft_r["layer1.0.conv2.weight"].norm().item() > 0
    assert ft_r["fc.weight"].norm().item() > 0

    opt_v = _forward_backward_with_hooks(tiny_vit, images, labels)
    ft_v = collect_fisher_tensors(tiny_vit, opt_v)
    assert ft_v["head.weight"].norm().item() > 0


def test_fisher_shape_matches_param(tiny_resnet, tiny_vit, dummy_cifar_batch):
    images, labels = dummy_cifar_batch
    for model in (tiny_resnet, tiny_vit):
        opt = _forward_backward_with_hooks(model, images, labels)
        ftensors = collect_fisher_tensors(model, opt)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert ftensors[name].shape == param.shape, f"shape mismatch for {name}"


# ---------------------------------------------------------------------------
# Layer registry -> chunk key resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_model,registry",
    [
        (build_resnet50_cifar, RESNET50_CIFAR_LAYERS),
        (build_vit_small_cifar, VIT_SMALL_LAYERS),
    ],
)
def test_all_registry_layers_resolve(build_model, registry):
    model = build_model()
    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    for label, layer_path in registry.items():
        keys = resolve_layer_chunk_keys(layer_path, chunk_specs_by_param)
        assert keys, f"registry entry '{label}' ({layer_path}) resolved to zero chunks"


def test_qkv_resolution_covers_all_three_heads():
    model = build_vit_small_cifar()
    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    keys = resolve_layer_chunk_keys("blocks.0.attn.qkv", chunk_specs_by_param)
    covered = {label for label in ("q", "k", "v") if any(f".{label}." in k for k in keys)}
    assert covered == {"q", "k", "v"}, f"missing q/k/v coverage: {covered}"


# ---------------------------------------------------------------------------
# JS-distance sanity
# ---------------------------------------------------------------------------


def test_js_distance_zero_for_identical_tensors():
    rng = np.random.default_rng(0)
    arr = np.abs(rng.normal(size=500)) + 1e-3
    assert js_distance_log(arr, arr) == pytest.approx(0.0, abs=1e-9)


def test_js_distance_positive_for_different_distributions():
    ref = np.ones(500)
    cur = np.random.default_rng(1).random(500) * 0.001
    d = js_distance_log(ref, cur)
    assert d is not None and d > 0.1


def test_js_distance_symmetric():
    rng = np.random.default_rng(2)
    a = np.abs(rng.normal(size=300)) + 1e-3
    b = np.abs(rng.normal(loc=2.0, size=300)) + 1e-3
    assert js_distance_log(a, b) == pytest.approx(js_distance_log(b, a), abs=1e-9)


# ---------------------------------------------------------------------------
# Freeze masking
# ---------------------------------------------------------------------------


def _setup_masks_and_freeze_one_chunk(model, optimizer):
    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=2, slice_mode="row"
    )
    param_train_masks = {
        name: torch.ones_like(p, dtype=torch.bool)
        for name, p in model.named_parameters()
        if name in chunk_specs_by_param
    }
    register_mask_hooks(model, param_train_masks)

    all_chunk_keys = {spec.key for specs in chunk_specs_by_param.values() for spec in specs}
    fc_specs = chunk_specs_by_param["fc.weight"]
    frozen_spec = fc_specs[0]
    selected_chunks = all_chunk_keys - {frozen_spec.key}

    apply_chunk_masks(model, optimizer, chunk_specs_by_param, selected_chunks, param_train_masks)
    return frozen_spec


def test_frozen_chunk_grad_is_zero_after_backward(tiny_resnet, dummy_cifar_batch):
    model = tiny_resnet
    images, labels = dummy_cifar_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    frozen_spec = _setup_masks_and_freeze_one_chunk(model, optimizer)

    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    grad = model.fc.weight.grad
    assert torch.all(grad[frozen_spec.index] == 0)
    assert torch.any(grad != 0), "expected non-frozen elements to still have gradient"


def test_frozen_chunk_optimizer_state_zeroed(tiny_resnet, dummy_cifar_batch):
    model = tiny_resnet
    images, labels = dummy_cifar_batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    frozen_spec = _setup_masks_and_freeze_one_chunk(model, optimizer)

    state = optimizer.state[model.fc.weight]
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.all(state[key][frozen_spec.index] == 0), f"{key} not zeroed on freeze"


# ---------------------------------------------------------------------------
# LayerMetricRecorder
# ---------------------------------------------------------------------------


class _FakeTrainer:
    def __init__(self, chunk_specs_by_param):
        all_keys = {spec.key for specs in chunk_specs_by_param.values() for spec in specs}
        self._chunk_js_history = {k: [(0, 0.1)] for k in all_keys}
        self._chunk_fisher_ema = {k: np.array([0.5, 0.5]) for k in all_keys}


@pytest.mark.parametrize(
    "build_model,registry",
    [
        (build_resnet50_cifar, RESNET50_CIFAR_LAYERS),
        (build_vit_small_cifar, VIT_SMALL_LAYERS),
    ],
)
def test_recorder_records_all_four_metrics_per_layer(build_model, registry, dummy_cifar_batch):
    model = build_model()
    images, labels = dummy_cifar_batch
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    recorder = LayerMetricRecorder(model, registry, chunk_specs_by_param)
    fake_trainer = _FakeTrainer(chunk_specs_by_param)
    recorder(step=0, model=model, trainer=fake_trainer)
    recorder(step=1, model=model, trainer=fake_trainer)

    history = recorder.as_dict()
    assert set(history.keys()) == {
        "fisher_drift", "fisher_magnitude", "grad_norm", "relative_update",
    }
    for metric_name, per_layer in history.items():
        assert set(per_layer.keys()) == set(registry.keys())
        for label, points in per_layer.items():
            assert len(points) == 2
            for step, value in points:
                assert np.isfinite(value), f"{metric_name}/{label} has non-finite value"


def test_relative_update_zero_on_first_call_then_positive(dummy_cifar_batch):
    model = build_resnet50_cifar()
    images, labels = dummy_cifar_batch
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    recorder = LayerMetricRecorder(model, RESNET50_CIFAR_LAYERS, chunk_specs_by_param)
    fake_trainer = _FakeTrainer(chunk_specs_by_param)

    recorder(step=0, model=model, trainer=fake_trainer)
    first_update = recorder.as_dict()["relative_update"]["fc"][0][1]
    assert first_update == 0.0

    with torch.no_grad():
        for p in model.fc.parameters():
            p.add_(0.1)

    recorder(step=1, model=model, trainer=fake_trainer)
    second_update = recorder.as_dict()["relative_update"]["fc"][1][1]
    assert second_update > 0.0


@pytest.mark.parametrize(
    "build_model,registry",
    [
        (build_resnet50_cifar, RESNET50_CIFAR_LAYERS),
        (build_vit_small_cifar, VIT_SMALL_LAYERS),
    ],
)
def test_recorder_raw_grads_match_layer_param_count(build_model, registry, dummy_cifar_batch):
    model = build_model()
    images, labels = dummy_cifar_batch
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    recorder = LayerMetricRecorder(model, registry, chunk_specs_by_param)
    fake_trainer = _FakeTrainer(chunk_specs_by_param)
    recorder(step=0, model=model, trainer=fake_trainer)
    recorder(step=5, model=model, trainer=fake_trainer)

    raw_grads = recorder.as_raw_grads()
    assert set(raw_grads.keys()) == set(registry.keys())
    for label, layer_path in registry.items():
        module = model.get_submodule(layer_path)
        expected_numel = sum(p.numel() for p in module.parameters(recurse=False))
        entry = raw_grads[label]
        assert entry["steps"] == [0, 5]
        assert entry["grads"].shape == (2, expected_numel)
        assert torch.isfinite(entry["grads"]).all()


def test_recorder_chunk_fisher_magnitude_history_covers_full_population(dummy_cifar_batch):
    model = build_resnet50_cifar()
    images, labels = dummy_cifar_batch
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    all_chunk_keys = {spec.key for specs in chunk_specs_by_param.values() for spec in specs}
    recorder = LayerMetricRecorder(model, RESNET50_CIFAR_LAYERS, chunk_specs_by_param)
    fake_trainer = _FakeTrainer(chunk_specs_by_param)
    recorder(step=0, model=model, trainer=fake_trainer)
    recorder(step=1, model=model, trainer=fake_trainer)

    magnitude_history = recorder.as_chunk_fisher_magnitude_history()
    # Full chunk population, not just the 7 curated RESNET50_CIFAR_LAYERS --
    # that's the whole point of tracking this separately from fisher_magnitude.
    assert set(magnitude_history.keys()) == all_chunk_keys
    for key, points in magnitude_history.items():
        assert len(points) == 2
        for step, value in points:
            assert np.isfinite(value), f"chunk {key} has non-finite magnitude"


def test_grad_norm_matches_manual_computation(dummy_cifar_batch):
    model = build_resnet50_cifar()
    images, labels = dummy_cifar_batch
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    param_to_modinfo = index_param_to_module(model)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model, param_to_modinfo, slice_blocks=4, slice_mode="row"
    )
    recorder = LayerMetricRecorder(model, RESNET50_CIFAR_LAYERS, chunk_specs_by_param)
    fake_trainer = _FakeTrainer(chunk_specs_by_param)
    recorder(step=0, model=model, trainer=fake_trainer)

    manual = torch.sqrt(
        sum(p.grad.pow(2).sum() for p in model.fc.parameters(recurse=False))
    ).item()
    recorded = recorder.as_dict()["grad_norm"]["fc"][0][1]
    assert recorded == pytest.approx(manual, rel=1e-5)
