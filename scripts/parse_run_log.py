"""Parse a FisherAdapTuneTrainer stdout log into structured epoch/freeze records.

Usage: python scripts/parse_run_log.py <log_file> [<log_file> ...]

Extracts, per log file:
  - each [Freeze] decision line (step, mean, std, threshold, frozen, remaining, params)
  - each Epoch line (train_loss, val loss, val accuracy)
and prints a compact table plus the pre/post-freeze accuracy delta for each freeze event,
so a new run's log can be digested in one glance instead of manually scanning stdout.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+): train_loss=([\d.]+)\s+(?:loss=([\d.]+)\s+accuracy=([\d.]+))?"
)
_FREEZE_DECISION_RE = re.compile(
    r"\[Freeze\] Step (\d+): variation mean=([\d.]+) std=([\d.]+) λ=(-?[\d.]+) "
    r"threshold=([\d.]+) \| frozen=(\d+) remaining=(\d+)"
)
_FREEZE_PARAMS_RE = re.compile(
    r"\[Freeze\] Step (\d+): trainable tensors=(\d+) chunks=(\d+) params=([\d,]+)"
)
_ZERO_FISHER_RE = re.compile(
    r"\[Freeze\] Step 0 zero-Fisher freeze: tensors=(\d+), chunks=(\d+), params=([\d,]+)"
)


def parse_log(path: Path) -> Dict:
    epochs: List[Dict] = []
    freeze_events: List[Dict] = []
    zero_fisher: Optional[Dict] = None
    pending_decision: Optional[Dict] = None

    for line in path.read_text().splitlines():
        m = _ZERO_FISHER_RE.search(line)
        if m:
            zero_fisher = {
                "tensors": int(m.group(1)),
                "chunks": int(m.group(2)),
                "params": int(m.group(3).replace(",", "")),
            }
            continue

        m = _FREEZE_DECISION_RE.search(line)
        if m:
            pending_decision = {
                "step": int(m.group(1)),
                "mean": float(m.group(2)),
                "std": float(m.group(3)),
                "lambda": float(m.group(4)),
                "threshold": float(m.group(5)),
                "frozen": int(m.group(6)),
                "remaining": int(m.group(7)),
            }
            continue

        m = _FREEZE_PARAMS_RE.search(line)
        if m and pending_decision is not None:
            pending_decision.update(
                {
                    "tensors": int(m.group(2)),
                    "chunks": int(m.group(3)),
                    "params": int(m.group(4).replace(",", "")),
                }
            )
            freeze_events.append(pending_decision)
            pending_decision = None
            continue

        m = _EPOCH_RE.search(line)
        if m:
            epoch, total, train_loss, val_loss, val_acc = m.groups()
            epochs.append(
                {
                    "epoch": int(epoch),
                    "total_epochs": int(total),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss) if val_loss else None,
                    "val_acc": float(val_acc) if val_acc else None,
                }
            )

    return {"zero_fisher": zero_fisher, "freeze_events": freeze_events, "epochs": epochs}


def render(path: Path, parsed: Dict) -> str:
    lines = [f"=== {path.name} ==="]
    zf = parsed["zero_fisher"]
    if zf:
        lines.append(
            f"  step0 zero-Fisher freeze: tensors={zf['tensors']} chunks={zf['chunks']} "
            f"params={zf['params']:,}"
        )

    epochs = parsed["epochs"]
    if epochs:
        lines.append(f"  epochs logged: {len(epochs)}/{epochs[0]['total_epochs']}")
        final = epochs[-1]
        lines.append(
            f"  final epoch {final['epoch']}: train_loss={final['train_loss']:.4f} "
            f"val_loss={final['val_loss']:.4f} val_acc={final['val_acc']:.4f}"
        )

    for ev in parsed["freeze_events"]:
        # accuracy immediately before/after this freeze step, by nearest epoch boundary
        before = [e for e in epochs if e.get("val_acc") is not None]
        pre_acc = next((e["val_acc"] for e in reversed(before)), None)
        lines.append(
            f"  [freeze @ step {ev['step']}] mean={ev['mean']:.4f} std={ev['std']:.4f} "
            f"λ={ev['lambda']:.3f} threshold={ev['threshold']:.4f} "
            f"chunks {ev['frozen']}->{ev['remaining']} kept "
            f"({100 * ev['remaining'] / max(1, ev['frozen'] + ev['remaining']):.1f}%) "
            f"params={ev['params']:,}"
        )

    if not parsed["freeze_events"]:
        lines.append("  (no [Freeze] decision events in this log -- nofreeze run or pre-event)")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"=== {path} === (not found)")
            continue
        parsed = parse_log(path)
        print(render(path, parsed))
        print()


if __name__ == "__main__":
    main()
