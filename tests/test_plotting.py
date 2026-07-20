"""Tests for src/utils/layers.py's full-chunk-population helpers and
src/utils/plotting.py's Appendix-B.1-style plots (JS-drift heatmap+spread,
tiered trajectories, drift-by-parameter-type violin). These operate on real
chunk structure from the tiny fixture models with synthetic JS values --
no training needed, keeps this in the fast suite (see test_fisher_metric.py
for the same tiny-model-plus-synthetic-data pattern).
"""

import numpy as np
import pytest

from src.fisher.fisher_core import build_all_chunk_specs, index_param_to_module
from src.models.resnet_cifar import build_resnet50_cifar
from src.models.vit_cifar import build_vit_small_cifar
from src.utils.layers import (
    classify_depth_group,
    classify_param_type,
    resolve_chunk_layer_and_block,
)
from src.utils.plotting import (
    plot_chunk_drift_heatmap,
    plot_chunk_drift_trajectories_by_tier,
    plot_drift_violin_by_group,
    plot_lambda_metric_sweep,
    plot_metric_evolution,
)

_DEPTH_GROUP_ORDER = {
    "resnet50": ["Stem", "Layer1-2 (early)", "Layer3-4 (late)", "FC / Head"],
    "vit_small": ["Patch Embed", "Blocks 0-2 (early)", "Blocks 3-5 (late)", "Head"],
}
_PARAM_TYPE_GROUP_ORDER = ["Norm", "Attn", "MLP-FC", "Other (Conv/FC)"]


def _tiny_chunk_specs(model_name):
    model = build_resnet50_cifar(layers=(1, 1, 1, 1)) if model_name == "resnet50" else build_vit_small_cifar(depth=2)
    param_to_modinfo = index_param_to_module(model)
    return build_all_chunk_specs(model, param_to_modinfo, slice_blocks=4, slice_mode="row")


def _synthetic_chunk_js_history(chunk_specs_by_param, n_points=5, seed=0):
    rng = np.random.default_rng(seed)
    all_keys = sorted({spec.key for specs in chunk_specs_by_param.values() for spec in specs})
    return {
        key: [(step * 10, float(rng.uniform(0.0, 1.0) + rng.normal(scale=0.02))) for step in range(n_points)]
        for key in all_keys
    }


# ---------------------------------------------------------------------------
# layers.py: chunk structural parsing + classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["resnet50", "vit_small"])
def test_resolve_chunk_layer_and_block_covers_all_chunks(model_name):
    chunk_specs_by_param, _ = _tiny_chunk_specs(model_name)
    all_keys = {spec.key for specs in chunk_specs_by_param.values() for spec in specs}
    chunk_layer, chunk_block = resolve_chunk_layer_and_block(chunk_specs_by_param)
    assert set(chunk_layer.keys()) == all_keys
    assert set(chunk_block.keys()) == all_keys
    for block_idx in chunk_block.values():
        assert block_idx is None or 0 <= block_idx < 4


@pytest.mark.parametrize(
    "layer_key,model_name,expected",
    [
        ("patch_embed.proj", "vit_small", "Patch Embed"),
        ("blocks.0.attn.qkv.q", "vit_small", "Blocks 0-2 (early)"),
        ("blocks.5.mlp.fc2", "vit_small", "Blocks 3-5 (late)"),
        ("norm", "vit_small", "Blocks 3-5 (late)"),
        ("head", "vit_small", "Head"),
        ("conv1", "resnet50", "Stem"),
        ("bn1", "resnet50", "Stem"),
        ("layer1.0.conv2", "resnet50", "Layer1-2 (early)"),
        ("layer4.2.conv3", "resnet50", "Layer3-4 (late)"),
        ("fc", "resnet50", "FC / Head"),
    ],
)
def test_classify_depth_group_known_layers(layer_key, model_name, expected):
    assert classify_depth_group(layer_key, model_name) == expected


@pytest.mark.parametrize(
    "layer_key,cls_name,expected",
    [
        ("blocks.0.norm1", "LayerNorm", "Norm"),
        ("layer1.0.bn2", "BatchNorm2d", "Norm"),
        ("blocks.0.attn.qkv.q", "Linear", "Attn"),
        ("blocks.0.attn.proj", "Linear", "Attn"),
        ("blocks.0.mlp.fc1", "Linear", "MLP-FC"),
        ("head", "Linear", "Other (Conv/FC)"),
        ("patch_embed.proj", "Conv2d", "Other (Conv/FC)"),
        ("layer1.0.conv2", "Conv2d", "Other (Conv/FC)"),
    ],
)
def test_classify_param_type_known_layers(layer_key, cls_name, expected):
    assert classify_param_type(layer_key, cls_name) == expected


# ---------------------------------------------------------------------------
# plotting.py: the 3 new figures render without crashing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["resnet50", "vit_small"])
def test_plot_chunk_drift_heatmap_renders(tmp_path, model_name):
    chunk_specs_by_param, _ = _tiny_chunk_specs(model_name)
    chunk_layer, chunk_block = resolve_chunk_layer_and_block(chunk_specs_by_param)
    history = _synthetic_chunk_js_history(chunk_specs_by_param)

    layer_block_values = {}
    for key, points in history.items():
        block_idx = chunk_block[key]
        if block_idx is None:
            continue
        layer_block_values.setdefault(chunk_layer[key], {})[block_idx] = points[-1][1]
    layer_group = {layer: classify_depth_group(layer, model_name) for layer in layer_block_values}

    out_path = tmp_path / "heatmap.png"
    plot_chunk_drift_heatmap(
        layer_block_values, layer_group, out_path, group_order=_DEPTH_GROUP_ORDER[model_name]
    )
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.parametrize("model_name", ["resnet50", "vit_small"])
def test_plot_chunk_drift_trajectories_by_tier_renders(tmp_path, model_name):
    chunk_specs_by_param, _ = _tiny_chunk_specs(model_name)
    history = _synthetic_chunk_js_history(chunk_specs_by_param)

    out_path = tmp_path / "tiers.png"
    plot_chunk_drift_trajectories_by_tier(history, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


@pytest.mark.parametrize("model_name", ["resnet50", "vit_small"])
def test_plot_drift_violin_by_group_renders(tmp_path, model_name):
    chunk_specs_by_param, chunk_to_class = _tiny_chunk_specs(model_name)
    chunk_layer, _ = resolve_chunk_layer_and_block(chunk_specs_by_param)
    history = _synthetic_chunk_js_history(chunk_specs_by_param)

    value_by_key = {k: pts[-1][1] for k, pts in history.items()}
    group_by_key = {
        k: classify_param_type(chunk_layer[k], chunk_to_class.get(k, "")) for k in value_by_key
    }
    out_path = tmp_path / "violin.png"
    plot_drift_violin_by_group(
        value_by_key, group_by_key, out_path, group_order=_PARAM_TYPE_GROUP_ORDER
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_functions_noop_on_empty_input(tmp_path):
    plot_chunk_drift_heatmap({}, {}, tmp_path / "empty_heatmap.png")
    plot_chunk_drift_trajectories_by_tier({}, tmp_path / "empty_tiers.png")
    plot_drift_violin_by_group({}, {}, tmp_path / "empty_violin.png")
    assert not (tmp_path / "empty_heatmap.png").exists()
    assert not (tmp_path / "empty_tiers.png").exists()
    assert not (tmp_path / "empty_violin.png").exists()


# ---------------------------------------------------------------------------
# plot_metric_evolution: freeze-step marker + accuracy overlay (report §4/§7)
# ---------------------------------------------------------------------------


def test_plot_metric_evolution_renders_with_freeze_marker_and_accuracy_overlay(tmp_path):
    history = {"layer.a": [(i * 100, 0.1 * i) for i in range(10)]}
    accuracy_history = [(i * 200, 0.4 + 0.02 * i) for i in range(5)]
    out_path = tmp_path / "fisher_drift.png"
    plot_metric_evolution(
        history, "fisher_drift", out_path, freeze_steps=[500], accuracy_history=accuracy_history
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_metric_evolution_renders_with_multiple_freeze_markers(tmp_path):
    # Multi-freeze configs (freeze_interval lowered, added 2026-07-20) fire
    # the threshold trigger more than once per run -- every marker must
    # appear, not just the first (this is exactly the gap found when
    # analyzing runs/vit_small_freeze_fix_meanjs_multifreeze/, whose 2nd,
    # much larger freeze event was invisible with a single freeze_step).
    history = {"layer.a": [(i * 100, 0.1 * i) for i in range(10)]}
    out_path = tmp_path / "fisher_drift.png"
    plot_metric_evolution(history, "fisher_drift", out_path, freeze_steps=[300, 600, 900])
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_metric_evolution_skips_freeze_markers_out_of_range(tmp_path):
    # --disable-freeze runs have no freeze events at all (empty list) -- must
    # not raise or draw a marker off the axis.
    history = {"layer.a": [(i * 100, 0.1 * i) for i in range(10)]}
    out_path = tmp_path / "fisher_drift.png"
    plot_metric_evolution(history, "fisher_drift", out_path, freeze_steps=[])
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_metric_evolution_without_optional_args_still_renders(tmp_path):
    history = {"layer.a": [(i * 100, 0.1 * i) for i in range(10)]}
    out_path = tmp_path / "grad_norm.png"
    plot_metric_evolution(history, "grad_norm", out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_lambda_metric_sweep: cross-run comparison figure
# ---------------------------------------------------------------------------


def test_plot_lambda_metric_sweep_renders(tmp_path):
    rows = [
        {"label": "nofreeze", "lambda": None, "chunk_selection_metric": "nofreeze",
         "pct_retained": 100.0, "accuracy_relative": 100.0},
        {"label": "freeze", "lambda": -1.0, "chunk_selection_metric": "total_variation",
         "pct_retained": 9.1, "accuracy_relative": 90.4},
        {"label": "fix_meanjs", "lambda": -1.0, "chunk_selection_metric": "mean_js",
         "pct_retained": 48.9, "accuracy_relative": 97.6},
    ]
    out_path = tmp_path / "sweep.png"
    plot_lambda_metric_sweep(rows, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_lambda_metric_sweep_noop_on_empty_input(tmp_path):
    out_path = tmp_path / "empty_sweep.png"
    plot_lambda_metric_sweep([], out_path)
    assert not out_path.exists()
