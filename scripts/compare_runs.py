"""Scan runs/*/summary.json and print a comparison table across experiments.

Usage: python scripts/compare_runs.py [runs_dir]  (default: runs/)
"""

import json
import sys
from pathlib import Path


def main() -> None:
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        data = json.loads(summary_path.read_text())
        cfg = data.get("config", {})
        rows.append(
            {
                "run": summary_path.parent.name,
                "model": data.get("model"),
                "freeze": not data.get("disable_freeze", True),
                "test_acc": data.get("test_accuracy"),
                "test_loss": data.get("test_loss"),
                "trainable": data.get("trainable_params"),
                "total": data.get("total_params"),
                "lambda": cfg.get("js_variance_lambda"),
                "freeze_interval": cfg.get("freeze_interval"),
                "fisher_ema_interval": cfg.get("fisher_ema_interval"),
                "min_freeze_param_numel": cfg.get("min_freeze_param_numel", 0),
                "patience": cfg.get("early_stopping_patience"),
            }
        )

    if not rows:
        print(f"No runs/*/summary.json found under {runs_dir}")
        return

    header = (
        f"{'run':45s} {'acc':>7s} {'loss':>7s} {'trainable':>10s} {'%params':>7s} "
        f"{'lambda':>7s} {'n(freeze)':>9s} {'m(ema)':>7s} {'numel_min':>9s} {'patience':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        pct = 100 * r["trainable"] / r["total"] if r["trainable"] and r["total"] else 0.0
        print(
            f"{r['run']:45s} {r['test_acc']:7.4f} {r['test_loss']:7.4f} "
            f"{r['trainable']:10,d} {pct:6.1f}% "
            f"{str(r['lambda']):>7s} {str(r['freeze_interval']):>9s} "
            f"{str(r['fisher_ema_interval']):>7s} {str(r['min_freeze_param_numel']):>9s} "
            f"{str(r['patience']):>8s}"
        )


if __name__ == "__main__":
    main()
