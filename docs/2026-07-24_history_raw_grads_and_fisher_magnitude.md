# Training history extended for future post-hoc metrics (2026-07-24)

User asked to adapt `src/` so a completed run's history is complete enough to plot *new* metrics
later — Fisher-based, and "orthogonalité dans l'espace de loss" (loss-space orthogonality) — by
reading the saved history file, without re-training. No new plotting function was requested or
built this session; the ask was specifically about making the history complete, not about the plot
itself.

## Key decisions, clarified with the user before implementing

Three things were genuinely open and would have meant rewriting twice if guessed wrong:

1. **What "orthogonality" means**: both temporal (cosine similarity between a layer's gradient at
   step t vs. t+Δ — drift in gradient *direction*, not just magnitude) and cross-layer (cosine
   similarity between two layers' gradients at the same step — interference/alignment). Either one
   needs the raw gradient vector, not a derived scalar computed during training — settling this
   first avoided building a narrower "just compute cosine-sim online" mechanism that couldn't
   answer the other half of the question.
2. **Storage format**: raw gradient vectors are too large for `history.json`'s existing pure-JSON
   convention (worst case ~9MB per layer per snapshot for ResNet50's larger conv layers) — go with
   a separate binary file (`torch.save`), not JSON-encoded floats.
3. **Granularity**: the 7 curated layers already tracked by `LayerMetricRecorder`/
   `RESNET50_CIFAR_LAYERS`/`VIT_SMALL_LAYERS`, not the full chunk population (hundreds of chunks) —
   keeps the new disk cost bounded to what the rest of the pipeline already treats as "the tracked
   layers."

## What got built

`src/utils/metrics.py::LayerMetricRecorder` gained two more things captured per call, at the same
`fisher_ema_interval` cadence as the existing 4 curated metrics — see the module's own updated
docstring for the full "why" on each:

- `self.raw_grads` / `as_raw_grads()`: per curated layer, a `(step, flat_grad_vector)` time series
  — the exact same `param.grad` already used for the existing `grad_norm` metric, kept instead of
  discarded once the norm is taken. `as_raw_grads()` stacks each layer's snapshots into one
  `Tensor[num_steps, num_params]` rather than saving a Python list of tensors.
- `self.chunk_fisher_magnitude_history` / `as_chunk_fisher_magnitude_history()`: Fisher-magnitude
  EMA for **every** trainable chunk the trainer tracks, not just the 7 curated layers — mirrors
  `trainer._chunk_js_history` (already a full-population time series) but for
  `trainer._chunk_fisher_ema`, which the trainer only ever overwrites in place with no history kept
  anywhere else in the existing code.

`src/fisher/trainer.py` was **not** touched again — both additions read fields already exposed on
the trainer instance passed into the step_callback, so the "one deliberate diff" invariant (see
`CLAUDE.md`) still holds; new metric evolution stays confined to `metrics.py`, per the original
plan's own rationale for passing `self` into the callback in the first place.

`src/train.py::_run_training` now writes `metrics_plots/raw_grads.pt`
(`torch.save(recorder.as_raw_grads(), ...)`) and `history.json` gained two keys:
`"raw_grads_file": "raw_grads.pt"` and `"chunk_fisher_magnitude_history"` (the latter inline, since
it's scalars).

## Real cost, flagged rather than hidden

At the validated `fisher_ema_interval=200`: a 40-epoch ResNet50 cluster run (`steps_per_epoch=351`,
~71 snapshots) now writes roughly **1.2GB** of `raw_grads.pt`, dominated by `layer4.2.conv2`
(3x3, 512 channels) and `layer4.2.conv3` (1x1 channel expansion) — the two largest curated tensors.
The ViT-small equivalent is much cheaper (~180MB — every curated layer there is far narrower,
`embed_dim=192`). `runs/` is entirely gitignored so this never touches git, but it's real new disk
usage per run on the cluster, worth knowing before launching a full batch. No opt-out knob (e.g. a
separate raw-grad-capture interval, or a way to disable it) was added, since none was requested —
flagging the size here rather than deciding for the user whether it's worth trading off.

## Tests

Fast suite 64 → 67 (new tests in `tests/test_fisher_metric.py`:
`test_recorder_raw_grads_match_layer_param_count`,
`test_recorder_chunk_fisher_magnitude_history_covers_full_population`). Both slow
`test_train_smoke` variants (ResNet50 + ViT, freeze + nofreeze) re-verified end to end:
`raw_grads.pt` written and loadable via `torch.load(..., weights_only=False)`, per-layer gradient
tensor shape matches that layer's own `parameters(recurse=False)` count, `history.json`'s two new
keys present.
