"""Tests for src/utils/logs.py -- recovering val accuracy and freeze-event
steps from a run's train.log (trainer.py keeps no in-memory history of
either, see that module's docstring)."""

from src.utils.logs import parse_freeze_steps, parse_val_accuracy_history

_SAMPLE_LOG = """\
[Val] Baseline validation before training...
[Val] Baseline: loss=2.3006  accuracy=0.1250
[Freeze] Zero-Fisher chunks found and frozen: 2
[Freeze] Step 0 zero-Fisher freeze: tensors=78, chunks=408, params=2,680,906
Epoch 1/40: train_loss=1.8042  loss=1.7604  accuracy=0.3752
Epoch 2/40: train_loss=1.5379  loss=1.5924  accuracy=0.4346
[Freeze] Step 4000: variation mean=0.4910 std=0.2248 λ=-1.300 threshold=0.1987 | frozen=22 remaining=386
[Freeze] Step 4000: trainable tensors=73 chunks=386 params=1,943,530
Epoch 3/40: train_loss=1.4195  loss=1.4186  accuracy=0.5025
[Freeze] Step 8000: mean_js mean=0.4513 std=0.1750 λ=-1.000 threshold=0.2763 | frozen=115 remaining=239
[Freeze] Step 8000: trainable tensors=49 chunks=239 params=23,770
Epoch 4/40: train_loss=1.3439  loss=1.4146  accuracy=0.4887
"""


def test_parse_val_accuracy_history(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text(_SAMPLE_LOG)
    history = parse_val_accuracy_history(log_path, steps_per_epoch=351)
    assert history == [
        (1 * 351, 0.3752),
        (2 * 351, 0.4346),
        (3 * 351, 0.5025),
        (4 * 351, 0.4887),
    ]


def test_parse_freeze_steps_excludes_step_zero_and_duplicate_lines(tmp_path):
    # Step 0's zero-Fisher freeze is a separate, unconditional mechanism (see
    # CLAUDE.md), not a js_variance_lambda threshold decision -- must not be
    # counted. Each real event also prints 2 lines at the same step (the
    # "variation mean=..."/"mean_js mean=..." decision line and a companion
    # "trainable tensors=..." line) -- must contribute exactly one entry.
    log_path = tmp_path / "train.log"
    log_path.write_text(_SAMPLE_LOG)
    assert parse_freeze_steps(log_path) == [4000, 8000]


def test_parse_functions_return_empty_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.log"
    assert parse_val_accuracy_history(missing, steps_per_epoch=351) == []
    assert parse_freeze_steps(missing) == []
