"""Recover per-epoch validation accuracy from a run's train.log.

FisherAdapTuneTrainer.fit() (src/fisher/trainer.py) never keeps an
in-memory history of validation accuracy across epochs -- it computes it
fresh each epoch, prints it, and (if wandb is active) logs it, then
discards it. Rather than adding a second history dict to trainer.py (the
class this project deliberately touches as little as possible, see
CLAUDE.md), the accuracy-vs-step overlay in plot_metric_evolution is fed by
re-reading the same train.log line train.py already writes -- the trainer's
own stdout is the only place this number survives.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

_EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+): train_loss=[\d.]+\s+loss=[\d.]+\s+accuracy=([\d.]+)")
_FREEZE_STEP_RE = re.compile(r"\[Freeze\] Step (\d+): (?:variation|mean_js) mean=")


def parse_val_accuracy_history(log_path: Path, steps_per_epoch: int) -> List[Tuple[int, float]]:
    """[(step, val_accuracy), ...], one point per completed epoch in the log.

    step = epoch_index * steps_per_epoch (the step count at which that
    epoch's end-of-epoch validation ran in fit()'s loop). Returns [] if
    log_path doesn't exist (older runs predating train.log, e.g.
    runs/vit_small_{freeze,nofreeze}/ -- no accuracy overlay for those).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    points: List[Tuple[int, float]] = []
    for line in log_path.read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            epoch = int(m.group(1))
            accuracy = float(m.group(3))
            points.append((epoch * steps_per_epoch, accuracy))
    return points


def parse_freeze_steps(log_path: Path) -> List[int]:
    """Steps at which a threshold-based freeze decision actually fired.

    Reads back the trainer's own "[Freeze] Step N: variation ..." / "...
    mean_js ..." print lines rather than computing multiples of
    freeze_interval from config -- config alone can't tell whether a later
    planned event was ever reached (early stopping can end the run first,
    as in runs/vit_small_freeze_fix_meanjs_multifreeze/, which never got to
    its 3rd planned event at step 12000). Excludes the separate one-time
    "Step 0 zero-Fisher freeze" line (a different, unconditional mechanism,
    not the js_variance_lambda threshold this marker is about -- see
    CLAUDE.md). Returns [] if log_path doesn't exist, same as
    parse_val_accuracy_history.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    steps = []
    for line in log_path.read_text().splitlines():
        m = _FREEZE_STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
    return steps
