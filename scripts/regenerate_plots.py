"""Regenerate a run's metrics_plots/ from its saved history.json, without
re-training -- lets a plotting-code change (new marker, new overlay, a
merged/renamed figure) apply to an already-completed run.

Only works for runs that have metrics_plots/history.json (added to
src/train.py alongside this script; runs completed before that change have
no raw per-step history anywhere on disk -- only the already-rendered PNGs
-- so they cannot be regenerated this way and must be re-run instead).

Usage: python scripts/regenerate_plots.py runs/<run_name> [runs/<run_name> ...]
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.fisher.fisher_core import build_all_chunk_specs, index_param_to_module
from src.train import _DEPTH_GROUP_ORDER, _PARAM_TYPE_GROUP_ORDER, build_model
from src.utils.layers import classify_depth_group, classify_param_type, resolve_chunk_layer_and_block
from src.utils.plotting import (
    plot_all_metrics_grid,
    plot_chunk_drift_heatmap,
    plot_chunk_drift_trajectories_by_tier,
    plot_drift_violin_by_group,
    plot_metric_evolution,
)


def regenerate(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics_plots"
    history_path = metrics_dir / "history.json"
    summary_path = run_dir / "summary.json"
    if not history_path.exists():
        print(f"skip {run_dir}: no metrics_plots/history.json (predates history persistence)")
        return

    history = json.loads(history_path.read_text())
    cfg = json.loads(summary_path.read_text())["config"]
    model_name = history["model"]
    freeze_step = history["freeze_step"]
    accuracy_history = [tuple(p) for p in history["accuracy_history"]]
    all_metrics = {
        metric: {label: [tuple(p) for p in points] for label, points in layers.items()}
        for metric, layers in history["metrics"].items()
    }
    chunk_js_history = {k: [tuple(p) for p in v] for k, v in history["chunk_js_history"].items()}
    chunk_to_class = history["chunk_to_class"]

    for metric_name, metric_history in all_metrics.items():
        plot_metric_evolution(
            metric_history, metric_name, metrics_dir / f"{metric_name}.png",
            freeze_step=freeze_step,
            accuracy_history=accuracy_history if metric_name == "fisher_drift" else None,
        )
    plot_all_metrics_grid(all_metrics, metrics_dir / "all_metrics_grid.png")

    # Chunk-population plots need chunk_specs_by_param (layer_key/block per
    # chunk key) -- deterministic from model architecture alone, so rebuild
    # an untrained model rather than persisting this in history.json too.
    model = build_model(model_name, cfg)
    chunk_specs_by_param, _ = build_all_chunk_specs(
        model,
        index_param_to_module(model),
        slice_blocks=cfg.get("fisher_slice_blocks", 4),
        slice_mode=cfg.get("fisher_slice_mode", "row"),
    )
    chunk_layer, chunk_block = resolve_chunk_layer_and_block(chunk_specs_by_param)
    chunk_final_js = {k: v[-1][1] for k, v in chunk_js_history.items() if v}

    layer_block_values = {}
    for chunk_key, js in chunk_final_js.items():
        layer = chunk_layer.get(chunk_key)
        block_idx = chunk_block.get(chunk_key)
        if layer is None or block_idx is None:
            continue
        layer_block_values.setdefault(layer, {})[block_idx] = js
    layer_group = {layer: classify_depth_group(layer, model_name) for layer in layer_block_values}
    plot_chunk_drift_heatmap(
        layer_block_values, layer_group, metrics_dir / "chunk_drift_heatmap.png",
        group_order=_DEPTH_GROUP_ORDER[model_name],
    )
    plot_chunk_drift_trajectories_by_tier(
        chunk_js_history, metrics_dir / "chunk_drift_trajectories_by_tier.png"
    )
    chunk_param_type = {
        k: classify_param_type(chunk_layer[k], chunk_to_class.get(k, ""))
        for k in chunk_final_js
        if k in chunk_layer
    }
    plot_drift_violin_by_group(
        chunk_final_js, chunk_param_type, metrics_dir / "chunk_drift_violin_by_param_type.png",
        group_order=_PARAM_TYPE_GROUP_ORDER, title="Drift by parameter type",
    )
    print(f"regenerated {run_dir}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        regenerate(Path(arg))


if __name__ == "__main__":
    main()
