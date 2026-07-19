"""Cross-run comparison plot for a chunk_selection_metric/js_variance_lambda
sweep: reads runs/*/summary.json directly (no history.json needed, unlike
scripts/regenerate_plots.py -- test_accuracy/trainable_params/config are
already in summary.json for every run regardless of when it ran), computes
% params retained and accuracy relative to the sweep's own nofreeze run, and
calls plot_lambda_metric_sweep. See
docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md §2 for the table
this mirrors as a figure.

Usage: python scripts/plot_sweep_summary.py <out.png> <nofreeze_run_dir> <freeze_run_dir> [<freeze_run_dir> ...]
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.plotting import plot_lambda_metric_sweep


def _load(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    out_path = Path(sys.argv[1])
    nofreeze_dir = Path(sys.argv[2])
    freeze_dirs = [Path(p) for p in sys.argv[3:]]

    nofreeze = _load(nofreeze_dir)
    nofreeze_acc = nofreeze["test_accuracy"]

    rows = [
        {
            "label": nofreeze_dir.name,
            "lambda": None,
            "chunk_selection_metric": "nofreeze",
            "pct_retained": 100.0 * nofreeze["trainable_params"] / nofreeze["total_params"],
            "accuracy_relative": 100.0,
        }
    ]
    for run_dir in freeze_dirs:
        data = _load(run_dir)
        cfg = data["config"]
        rows.append(
            {
                "label": run_dir.name,
                "lambda": cfg.get("js_variance_lambda"),
                "chunk_selection_metric": cfg.get("chunk_selection_metric", "total_variation"),
                "pct_retained": 100.0 * data["trainable_params"] / data["total_params"],
                "accuracy_relative": 100.0 * data["test_accuracy"] / nofreeze_acc,
            }
        )

    plot_lambda_metric_sweep(rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
