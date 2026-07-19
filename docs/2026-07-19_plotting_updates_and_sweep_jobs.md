# Plotting updates + overnight sweep jobs, following up on the 5-experiments report (2026-07-19)

Implements the actionable items from
`docs/2026-07-19_vit_freeze_tuning_5_experiments_report.md` §7 ("Pistes d'amélioration explorées
et à explorer"). Full rationale for each SLURM job lives in `slurm/README.md`'s "Overnight batch"
section; this note covers the plotting-code changes and one important limitation discovered while
implementing "regenerate all existing runs' plots."

## What changed in `src/utils/plotting.py` / `src/train.py`

- **`freeze_step` marker** on all 4 curated per-run plots (`fisher_drift`, `fisher_magnitude`,
  `grad_norm`, `relative_update`): a vertical dotted line at the one-shot freeze step. Directly
  addresses report §4: `fisher_drift`/`fisher_magnitude` can jump right after this step purely
  because the trainer's per-layer average is recomputed over a smaller surviving-chunk group
  (frozen chunks are dropped, not zeroed) — not a real behavior change. The marker doesn't remove
  the artifact (still need `relative_update`/`grad_norm`/the `chunk_*` plots to know what's really
  frozen, per the report), it just flags the step so a reader isn't misled.
- **`accuracy_history` overlay**, `fisher_drift.png` only (as asked — "direct sur la courbe
  fisher_drift.png"), on a right-hand twin axis. Source: `src/utils/logs.py`'s
  `parse_val_accuracy_history`, which re-reads the run's own `train.log` rather than adding a
  second history dict to `src/fisher/trainer.py` — that file is deliberately touched as little as
  possible (see CLAUDE.md), and `fit()` already prints every epoch's val accuracy; nothing there
  needed to change.
- **`plot_lambda_metric_sweep`**: cross-run comparison (% params retained vs. accuracy relative to
  that sweep's nofreeze run, one point per config, marker = `chunk_selection_metric`, label = λ).
  The one deliberately metric-vs-metric figure in this module — it answers "which config is worth
  keeping," a different question from every other plot here, which is why it's a separate function
  rather than an option on `plot_metric_evolution`. Generated now for the 5 existing ViT sweep runs
  via `scripts/plot_sweep_summary.py` → `runs/sweep_summaries/vit_lambda_metric_sweep.png` (matches
  the table in the 5-experiments report §2); ready to rerun for ResNet50 once tonight's sweep lands.

**Redundancy check** (as asked): compared `all_metrics_grid.png` against the 4 individual curated
PNGs it grids together — same underlying series, but judged not truly redundant (one is a
single-glance overview, the other 4 are what the 5-experiments report actually cites for detailed
reading, and per that report's own §1 each of the 4 metrics resolves a distinct ambiguity the
others leave open). Left as-is. `chunk_drift_*` plots untouched per direct instruction.

## Why the 7 already-completed runs' plots can't actually be regenerated

Asked to "relance tous les plottings des runs actuels" with the new marker/overlay. Checked what's
on disk for the 7 existing run directories (`vit_small_{freeze,nofreeze}`,
`vit_small_freeze_fix_{lambda13,lambda16,meanjs}`, `resnet50_{freeze,nofreeze}`): each has
`summary.json`, `train.log`/a SLURM `.out` file, checkpoints, and the already-rendered PNGs — but
**no raw per-step or per-chunk history was ever persisted**. Every curve in every PNG (Fisher
drift, magnitude, grad norm, relative update, and the 3 `chunk_*` figures) only ever existed in
memory during training; `LayerMetricRecorder.history` and `trainer._chunk_js_history` were read
once at the end of `run_training` to draw the PNGs, then discarded. There is nothing to re-plot
from — only full re-training reproduces the underlying series (the two `vit_small_{freeze,
nofreeze}` baseline runs don't even have a log file to pull an accuracy curve from).

Fix going forward: `run_training` now also writes `metrics_plots/history.json` (raw
`LayerMetricRecorder` history + `trainer._chunk_js_history`/`_chunk_to_class` + the accuracy
history + freeze step), and `scripts/regenerate_plots.py runs/<run_name>` rebuilds all 8 PNGs from
it without re-training. Verified end-to-end on a synthetic smoke-test run (train → history.json
written → `regenerate_plots.py` → all 8 PNGs re-rendered, byte-different from the originals as
expected). This only helps runs completed **after** this change — every run in tonight's SLURM
batch will have it; the 7 pre-existing runs do not, and short of re-running them (which the
overnight batch does for ResNet50 across a fresh λ/metric sweep anyway, plus 2 new ViT points),
their plots stay exactly as already rendered.

## Overnight SLURM batch

6 new jobs (2 ViT, 4 ResNet50) — full list, rationale, and submission commands in
`slurm/README.md`'s "Overnight batch, 2026-07-19" section. Not repeated here to avoid the two
copies drifting; that file is the one to check before submitting.
