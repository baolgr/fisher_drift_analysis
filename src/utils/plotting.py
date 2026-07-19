"""Metric-evolution plots: x = training iteration, y = metric value, one
line per layer. Never a metric-vs-metric scatter -- that's the whole point
of this module, per the advisor's explicit ask.

Also includes full-chunk-population summary views modeled on the
FisherAdapTune paper's Appendix B.1 (Figures 7/8, "Fisher Drift
Observations"): a layer x column-block JS-distance heatmap with a
within-layer-spread sidebar, drift trajectories grouped by tier, and a
drift-by-component violin. These summarize *one* metric (JS-distance)
across structural axes (layer, column-block, component group) -- still
never one metric plotted against another.

Kept separate from src/fisher/utils.py (the copied library, which stays
generic/model-agnostic, matching the original FisherAdapTune repo).

Color choices below follow the house dataviz method (color = the entity's
job: sequential for magnitude, one-hue-monotone for an ordinal tier, fixed-
order categorical for nominal groups) using this project's validated
palette. The categorical sidebar in plot_chunk_drift_heatmap deliberately
skips the palette's blue slot (reserved for that figure's sequential scale)
to avoid one hue meaning two different things in the same figure.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

_METRIC_TITLES = {
    "fisher_drift": "Fisher drift (JS-distance)",
    "fisher_magnitude": "Fisher magnitude",
    "grad_norm": "Gradient L2 norm",
    "relative_update": "Relative weight update ||dw|| / ||w||",
}

# Validated palette (dataviz skill, references/palette.md) -- documented
# hex only, no eyeballed values.
_SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_CATEGORICAL_SIDEBAR = ["#008300", "#e87ba4", "#eda100", "#1baf7a"]  # slots 2/3/4/5 -- blue reserved for the heatmap's sequential scale
_CATEGORICAL_VIOLIN = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # documented "first four", validated all-pairs both modes
_ORDINAL_TIER = {"High": "#184f95", "Mid": "#3987e5", "Low": "#86b6ef"}  # one hue (blue), monotone lightness, light end clears the 2:1 ordinal floor
_MISSING_COLOR = "#f0efec"  # neutral gray midpoint, for heatmap cells with no data


def _assign_group_colors(group_order: List[str], palette: List[str]) -> Dict[str, str]:
    if len(group_order) > len(palette):
        raise ValueError(
            f"{len(group_order)} groups but only {len(palette)} validated colors -- "
            "fold extra groups into an existing one rather than cycling the palette."
        )
    return {group: palette[i] for i, group in enumerate(group_order)}


def plot_metric_evolution(
    metric_history: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    out_path: Path,
    log_scale_y: bool = False,
    freeze_step: Optional[int] = None,
    accuracy_history: Optional[List[Tuple[int, float]]] = None,
) -> None:
    """One figure: x = iteration, y = metric_name, one line per layer.

    freeze_step draws a vertical marker at the one-shot freeze decision step
    (see CLAUDE.md/plan.md: freeze is a single trigger, not gradual) -- added
    per docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md §4/§7: a
    layer's curve can jump right after this step purely because the
    trainer's per-layer average is recomputed over a smaller surviving-chunk
    group (frozen chunks are dropped, not zeroed), not because the layer's
    own behavior actually changed. The marker doesn't fix the artifact
    (still requires relative_update/grad_norm or the chunk_* plots to know
    what's really frozen, see that report), it just flags the step so a
    reader doesn't mistake post-marker jumps for a real acceleration. Silently
    skipped if freeze_step falls outside the plotted step range (nofreeze
    runs pass a freeze_step far beyond num_epochs*steps_per_epoch).

    accuracy_history (val accuracy vs step, from the same run) overlays on a
    right-hand twin axis -- lets a reader see the freeze-shock (report §2)
    directly against the drift curve it's supposedly caused by, rather than
    cross-referencing a separate epoch table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in metric_history.items():
        if not points:
            continue
        steps, values = zip(*points)
        ax.plot(steps, values, label=label, marker="o", markersize=2, linewidth=1)

    ax.set_xlabel("Iteration (step)")
    ax.set_ylabel(_METRIC_TITLES.get(metric_name, metric_name))
    ax.set_title(_METRIC_TITLES.get(metric_name, metric_name))
    if log_scale_y:
        ax.set_yscale("log")

    all_steps = [s for points in metric_history.values() for s, _ in points]
    if accuracy_history:
        all_steps += [s for s, _ in accuracy_history]
    if freeze_step is not None and all_steps and min(all_steps) <= freeze_step <= max(all_steps):
        ax.axvline(freeze_step, color="#52514e", linestyle=":", linewidth=1.3, zorder=0)
        ax.text(
            freeze_step, 1.0, " freeze", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=7, color="#52514e",
        )

    lines, labels = ax.get_legend_handles_labels()
    if accuracy_history:
        ax2 = ax.twinx()
        acc_steps, acc_values = zip(*accuracy_history)
        (acc_line,) = ax2.plot(
            acc_steps, acc_values, color="#0b0b0b", linestyle="--", linewidth=1.8,
            marker="s", markersize=3, label="Val accuracy (right axis)",
        )
        ax2.set_ylabel("Val accuracy")
        ax2.set_ylim(0, 1)
        lines += [acc_line]
        labels += [acc_line.get_label()]

    ax.legend(lines, labels, fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_metrics_grid(
    all_metrics: Dict[str, Dict[str, List[Tuple[int, float]]]],
    out_path: Path,
    layer_label_order: Optional[List[str]] = None,
) -> None:
    """Grid of subplots, one per metric, each metric-vs-iteration with one line per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = list(all_metrics.keys())
    n = len(metric_names)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for i, metric_name in enumerate(metric_names):
        ax = axes[i // n_cols][i % n_cols]
        metric_history = all_metrics[metric_name]
        labels = layer_label_order if layer_label_order is not None else list(metric_history.keys())
        for label in labels:
            points = metric_history.get(label, [])
            if not points:
                continue
            steps, values = zip(*points)
            ax.plot(steps, values, label=label, marker="o", markersize=2, linewidth=1)
        ax.set_xlabel("Iteration (step)")
        ax.set_ylabel(_METRIC_TITLES.get(metric_name, metric_name))
        ax.set_title(_METRIC_TITLES.get(metric_name, metric_name))
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_chunk_drift_heatmap(
    layer_block_values: Dict[str, Dict[int, float]],
    layer_group: Dict[str, str],
    out_path: Path,
    num_blocks: int = 4,
    group_order: Optional[List[str]] = None,
    value_label: str = "Fisher drift (JS-distance, EMA)",
    title: str = "Final JS per column block (layers sorted by mean)",
) -> None:
    """Paper Fig 7(a)/8(a): layer x column-block heatmap (rows sorted by
    mean, highest at top) with a component-group sidebar and a within-layer
    spread (max-min) bar, both sharing the heatmap's row order.

    layer_block_values: layer_key -> {block_idx: final EMA-JS value}, e.g.
    from resolve_chunk_layer_and_block + a chunk's last _chunk_js_history
    entry. layer_group: layer_key -> component label (e.g. classify_depth_group).
    Rows with fewer than num_blocks values (QKV chunks split rows 3-way
    before block-slicing, or an uneven axis_size) show gaps, not zeros.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not layer_block_values:
        return

    def row_mean(layer: str) -> float:
        vals = list(layer_block_values[layer].values())
        return float(np.mean(vals)) if vals else float("-inf")

    layers = sorted(layer_block_values, key=row_mean, reverse=True)
    matrix = np.full((len(layers), num_blocks), np.nan)
    for i, layer in enumerate(layers):
        for block_idx, value in layer_block_values[layer].items():
            if 0 <= block_idx < num_blocks:
                matrix[i, block_idx] = value
    spread = np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)

    groups = group_order if group_order is not None else sorted(set(layer_group.values()))
    group_color = _assign_group_colors(groups, _CATEGORICAL_SIDEBAR)
    sidebar = np.array([[groups.index(layer_group.get(layer, groups[0]))] for layer in layers])
    sidebar_cmap = ListedColormap([group_color[g] for g in groups])

    # 4 explicit columns (sidebar, heatmap, colorbar, spread) rather than
    # letting fig.colorbar(ax=...) dynamically steal space from ax_heat --
    # with 3 pre-allocated axes, the auto-inserted colorbar axis collided
    # with ax_spread instead of respecting the gridspec's width_ratios.
    fig, axes = plt.subplots(
        1, 4, figsize=(10, max(4, 0.03 * len(layers))),
        gridspec_kw={"width_ratios": [0.4, 6, 0.4, 1.4], "wspace": 0.25},
    )
    ax_side, ax_heat, ax_cbar, ax_spread = axes
    ax_side.sharey(ax_heat)
    ax_spread.sharey(ax_heat)

    ax_side.imshow(sidebar, aspect="auto", cmap=sidebar_cmap, vmin=0, vmax=len(groups) - 1)
    ax_side.set_xticks([])
    ax_side.set_yticks([])

    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_BLUE).with_extremes(bad=_MISSING_COLOR)
    im = ax_heat.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap)
    ax_heat.set_xticks(range(num_blocks))
    ax_heat.set_xticklabels([f"Col {b}" for b in range(num_blocks)])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Column block")
    ax_heat.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label(value_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    y = np.arange(len(layers))
    ax_spread.barh(y, np.nan_to_num(spread), color="#e34948", height=0.9)
    ax_spread.invert_yaxis()
    ax_spread.set_xlabel("Within-layer\nspread (max-min)", fontsize=8)
    ax_spread.yaxis.set_visible(False)

    legend_handles = [Patch(facecolor=group_color[g], label=g) for g in groups]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=len(groups), fontsize=7, frameon=False,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chunk_drift_trajectories_by_tier(
    chunk_history: Dict[str, List[Tuple[int, float]]],
    out_path: Path,
    top_k: int = 6,
    bottom_k: int = 6,
    mid_k: int = 4,
    title: str = "JS distance trajectories by drift tier",
) -> None:
    """Paper Fig 8(b): individual chunk trajectories grouped into High/Mid/Low
    drift tiers by final value, legend keyed by tier (not by the hundreds of
    individual chunk names -- same reason the paper's own legend is tiered).
    Tiers are ordinal (High > Mid > Low is a real order), so they share one
    hue at monotone lightness rather than three unrelated categorical colors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    finals = {k: pts[-1][1] for k, pts in chunk_history.items() if pts}
    if not finals:
        return
    ranked = sorted(finals, key=lambda k: finals[k], reverse=True)

    n = len(ranked)
    top_k = min(top_k, n)
    bottom_k = min(bottom_k, max(0, n - top_k))
    remaining = ranked[top_k: n - bottom_k]
    mid_k = min(mid_k, len(remaining))
    mid_idx = np.linspace(0, len(remaining) - 1, mid_k, dtype=int) if mid_k and remaining else []
    tiers = {
        "High": ranked[:top_k],
        "Mid": [remaining[i] for i in mid_idx],
        "Low": ranked[n - bottom_k:],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    linestyle = {"High": "-", "Mid": "--", "Low": "-"}
    for tier, keys in tiers.items():
        for key in keys:
            steps, values = zip(*chunk_history[key])
            ax.plot(
                steps, values, color=_ORDINAL_TIER[tier], linestyle=linestyle[tier],
                linewidth=1.2, alpha=0.85,
            )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Fisher drift (JS-distance, EMA)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    handles = [
        Line2D([0], [0], color=_ORDINAL_TIER[t], linestyle=linestyle[t], linewidth=1.5,
               label=f"{t}-drift (n={len(tiers[t])})")
        for t in ("High", "Mid", "Low") if tiers[t]
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drift_violin_by_group(
    value_by_key: Dict[str, float],
    group_by_key: Dict[str, str],
    out_path: Path,
    group_order: Optional[List[str]] = None,
    value_label: str = "Final Fisher drift (JS-distance, EMA)",
    title: str = "Drift by parameter type",
) -> None:
    """Paper Fig 7(c)/8(c): violin of a per-chunk scalar grouped by a nominal
    category (e.g. classify_param_type), one mean-value label per group like
    the paper's own annotated means.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_groups = group_order if group_order is not None else sorted(set(group_by_key.values()))
    group_color = _assign_group_colors(all_groups, _CATEGORICAL_VIOLIN)
    data_by_group = {g: [] for g in all_groups}
    for key, value in value_by_key.items():
        group = group_by_key.get(key)
        if group in data_by_group:
            data_by_group[group].append(value)
    groups = [g for g in all_groups if data_by_group[g]]
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(1, len(groups) + 1)
    datasets = [data_by_group[g] for g in groups]
    parts = ax.violinplot(datasets, positions=positions, showmeans=True, showextrema=True)
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(group_color[g])
        pc.set_edgecolor(group_color[g])
        pc.set_alpha(0.35)
    for part_name in ("cmeans", "cmaxes", "cmins", "cbars"):
        parts[part_name].set_edgecolor("#52514e")
        parts[part_name].set_linewidth(1)

    rng = np.random.default_rng(0)
    for pos, g in zip(positions, groups):
        values = data_by_group[g]
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            [pos + j for j in jitter], values, color=group_color[g], s=10, alpha=0.6,
            edgecolors="none",
        )
        mean_val = float(np.mean(values))
        ax.text(
            pos, max(values) + 0.03 * (max(values) - min(values) + 1e-6), f"{mean_val:.2f}",
            ha="center", fontsize=8, color="#0b0b0b",
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{g}\n(n={len(data_by_group[g])})" for g in groups])
    ax.set_ylabel(value_label)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lambda_metric_sweep(
    rows: List[Dict],
    out_path: Path,
    title: str = "Freeze sweep: accuracy vs. params retained",
) -> None:
    """Cross-run comparison for a chunk_selection_metric/js_variance_lambda
    sweep (one point per completed run), modeled on the table in
    docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md §2/§7 --
    x = % trainable params retained, y = accuracy relative to that sweep's
    nofreeze baseline, marker shape = chunk_selection_metric, point label =
    lambda. Unlike every other plot in this module this is metric-vs-metric
    by necessity (it summarizes *runs*, not one run's iteration history) --
    it answers a different question ("which config is worth keeping") than
    the per-run curves, which is why it lives in its own function rather
    than feeding plot_metric_evolution.

    rows: list of dicts with keys label, lambda (float or None for
    nofreeze), chunk_selection_metric, pct_retained (0-100),
    accuracy_relative (0-100, vs. the sweep's own nofreeze run).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    metric_marker = {"total_variation": "o", "mean_js": "^", "nofreeze": "*"}
    metric_color = {"total_variation": "#2a78d6", "mean_js": "#e87ba4", "nofreeze": "#52514e"}

    fig, ax = plt.subplots(figsize=(7, 5))
    seen_metrics = []
    for row in rows:
        metric = row.get("chunk_selection_metric") or "nofreeze"
        if metric not in seen_metrics:
            seen_metrics.append(metric)
        ax.scatter(
            row["pct_retained"], row["accuracy_relative"],
            marker=metric_marker.get(metric, "o"), color=metric_color.get(metric, "#2a78d6"),
            s=70, zorder=3,
        )
        lam = row.get("lambda")
        point_label = row["label"] if lam is None else f"{row['label']}\n(λ={lam:g})"
        ax.annotate(
            point_label, (row["pct_retained"], row["accuracy_relative"]),
            textcoords="offset points", xytext=(6, 6), fontsize=7,
        )

    ax.axhline(100.0, color="#52514e", linestyle=":", linewidth=1, zorder=0)
    ax.set_xlabel("Trainable params retained (%)")
    ax.set_ylabel("Test accuracy relative to nofreeze (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=metric_marker.get(m, "o"), color=metric_color.get(m, "#2a78d6"),
               linestyle="", markersize=8, label=m)
        for m in seen_metrics
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
