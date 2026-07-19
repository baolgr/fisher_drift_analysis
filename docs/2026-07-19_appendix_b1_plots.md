# Appendix B.1-style plots added to plotting.py (2026-07-19)

User asked to reproduce the 3 plot types in the paper's Appendix B.1 ("Fisher Drift Observations",
Figures 7/8: SAM2-Tiny and SegFormer-B3) into `src/utils/plotting.py`, over the full chunk
population rather than the curated 7-layer registry, plus explain what `relative_update.png`/
`grad_norm.png` are for. No training was run for this (code + tests only) — see below for how to
actually produce these PNGs for a real run.

## Paper methodology cross-check (docs/paper_fisheradaptune.pdf, pp.14-20)

- **A.2**: the plotted quantity is Jensen-Shannon *distance* (sqrt of JS divergence) between
  log-transformed, shared-bin-histogram PMFs of consecutive Fisher snapshots — confirmed
  byte-for-byte against `src/fisher/fisher_core.py`'s `js_distance_log`/`_build_histogram_log`
  (unmodified copy, per the hard rule).
- **Appendix C, Algorithm 2** gives the precise per-quantity formulas, which resolved an ambiguity:
  step 9's raw consecutive drift `d̄` gets EMA-smoothed at step 10 into `d̂` (this is what the
  caption's `EMA[d_JS(...)]` refers to, and it's exactly `trainer._chunk_js_history`'s stored
  value — confirmed by reading `update_chunk_js_history` in fisher_core.py: `curr_js_ema = js_decay *
  prev_js_ema + (1-js_decay) * js`, stored straight into `chunk_js_history`). Step 11 computes a
  *separate* running mean `ď` over accumulated history — that's `chunk_selection_metric: mean_js`,
  not what this repo's configs use (`total_variation`, a cumulative sum of absolute deltas — not
  described anywhere in the paper's algorithm; already flagged as a discrepancy in
  `docs/fisheradaptune_freeze_experiments.md`'s "Open threads"). **The new plots read
  `_chunk_js_history` directly**, so they show the paper-faithful `EMA[JS]` quantity regardless of
  which `chunk_selection_metric` a given run used for its freeze *decision*.
- **Algorithm 2 step 7**: "partition each layer's Fisher tensor into *k* column-wise parameter
  groups" — this is exactly `fisher_slice_blocks` (both configs use 4), confirming Fig 7(a)'s
  "Col 0/1/2/3" x-axis maps directly onto this repo's existing chunk-block structure. No new
  chunking concept was needed — `plot_chunk_drift_heatmap`'s x-axis is literally the block index
  already encoded in each `ChunkSpec.key`'s `rowblockN`/`colblockN`/QKV `blockN` suffix (parsed back
  out by the new `resolve_chunk_layer_and_block` in `layers.py`, since `ChunkSpec` itself has no
  numeric block field).

## What got built

`src/utils/layers.py`: `resolve_chunk_layer_and_block` (chunk key -> layer_key + column-block index,
pure string parsing) and two classifiers, `classify_depth_group` (4-way, per model: Patch
Embed/Stem, early blocks, late blocks, Head/FC — mirrors the paper's Neck/EarlyBlk/LateBlk/Decoder
sidebar) and `classify_param_type` (Norm/Attn/MLP-FC/Other — mirrors Fig 8(c)). Both are pure
functions over `(layer_key, ...)` strings, no model object needed at plot time.

`src/utils/plotting.py`: three new functions, colors from the dataviz skill's validated palette
(sequential blue for the heatmap's JS-distance scale; the documented "first four" categorical slots
for nominal groups; one hue at monotone lightness for the High/Mid/Low drift tiers, since tier order
is ordinal, not nominal — see the module's updated docstring for the full rationale, including why
the heatmap's sidebar deliberately skips the palette's blue slot).

- `plot_chunk_drift_heatmap`: layer x column-block heatmap (rows sorted by mean, highest at top),
  component-group sidebar, within-layer-spread bars — combines paper panels (a)+(c) into one figure
  like the paper's own Fig 7 layout.
- `plot_chunk_drift_trajectories_by_tier`: individual trajectories for the top/bottom-K chunks by
  final drift plus a few evenly-spaced mid-rank ones, legend keyed by tier not by chunk name (paper
  Fig 8(b)).
- `plot_drift_violin_by_group`: violin + jittered strip + mean label per group (paper Fig 7(c)/8(c)).
  Called with `classify_param_type` groups — directly testable against the *already-documented*
  v1 finding in `docs/fisheradaptune_freeze_experiments.md` ("bias/LayerNorm chunks systematically
  outscore weight-matrix chunks") next time a real run produces this plot.

`train.py`: new `_plot_fisher_group_summaries(trainer, model_name, metrics_dir)`, called once after
`trainer.fit()` alongside the existing per-layer plots. Reads `trainer._chunk_js_history` /
`_chunk_specs_by_param` / `_chunk_to_class` directly (same "reach into trainer internals from
train.py" pattern already used for `_param_train_masks`), so no `src/fisher/trainer.py` change was
needed — the existing `step_callback` extension point wasn't the right fit here (these are
end-of-training summaries over *all* chunks, not a per-step incremental recording).

## Bug the tests caught (see CLAUDE.md's classifiers entry)

`cls_token`/`pos_embed` (raw `nn.Parameter`s, no owning Conv2d/Linear/Norm submodule) still get
`ChunkSpec`s from `build_all_chunk_specs` — it doesn't filter by module type, just
`param.requires_grad`. They're invisible to `AdaFisherBackbone`'s hooks (so their actual Fisher
values are zero, and step-0 zero-Fisher freeze likely prunes them out of `_chunk_js_history` almost
immediately in a real run) but `tests/test_plotting.py`'s synthetic-history tests build history for
every chunk regardless, surfacing the classification gap directly:
`classify_depth_group("cls_token"/"pos_embed", "vit_small")` fell through to `"Other"`, which isn't
in the declared 4-color group order, and the heatmap's sidebar hard-crashed. Fixed by folding both
into `"Patch Embed"`. Verified no other layer_key falls through to `"Other"` for either model via a
one-off enumeration script (both models' full chunk populations, not just the tiny fixtures).

## What the real output looks like

Generated a real (non-training-run, synthetic-dataloader) preview via the same path
`test_train_smoke` exercises, inspected all 3 PNGs directly:

- Heatmap: renders correctly, blue sequential scale, 4-color sidebar, red spread bars, colorbar in
  its own gridspec column. Needed one layout fix: `fig.colorbar(im, ax=ax_heat)` dynamically steals
  space and collided with the pre-allocated spread-bar axis once `fig.tight_layout()` was removed (to
  fix an unrelated deprecation warning) — fixed by giving the colorbar its own explicit 4th gridspec
  column (`cax=`) instead of letting it auto-insert.
- Tiered trajectories: clean fan-out from origin, dark-to-light blue by tier, legend with per-tier n.
- Violin: 4 clearly separated groups, jittered points, mean labels matching the paper's own annotated
  style.

Not regenerated for a real CIFAR run this session (no training was run — code + tests only, per the
user's ask). Next real `python src/train.py --model vit_small ...` run will produce
`runs/<name>/metrics_plots/chunk_drift_heatmap.png` / `chunk_drift_trajectories_by_tier.png` /
`chunk_drift_violin_by_param_type.png` alongside the existing 5 PNGs, worth a fresh look then
(the synthetic preview used 2 epochs of random data, not representative of real drift structure).

## relative_update / grad_norm — what they're for (user question)

The paper's own Fisher-only pipeline has no equivalent of either — this repo's rationale
(`src/utils/metrics.py`, `CLAUDE.md`'s "4 metrics" entry) is that JS-drift alone is ambiguous (a
flat/low drift curve can't distinguish "converged" from "never had signal"), and Fisher magnitude
only partly resolves that ambiguity since it's still Fisher-derived and shares the same
histogram/chunking-specific machinery (and its potential biases, e.g. the small-tensor
histogram-noise bias already documented for JS-distance in the freeze-tuning log).

- **`grad_norm`** (`‖∇W L‖`, read right after `backward()`, before any optimizer step): a
  Fisher-independent diagnostic of raw optimization pressure on a layer *right now*. Its value is
  precisely that it can't share systematic biases with the JS-distance computation — no histograms,
  no log-transform, no chunk-size sensitivity. When grad_norm and Fisher-drift agree (both say a
  layer is inactive), that's real corroboration from an independent signal, not just one metric
  echoing itself.
- **`relative_update`** (`‖Δw‖ / ‖w‖` over the metric-recording interval): what AdamW's adaptive step
  actually did to the weights, as opposed to the raw gradient signal before momentum/second-moment
  normalization. A layer can have a large `grad_norm` but a tiny realized `relative_update` (if its
  gradient has been consistently large, AdamW's `v` estimate grows and throttles the effective step),
  so the two together separate "how much pressure" from "how much actually moved."
- Concretely useful already (see `docs/2026-07-19_vit_v5_curve_analysis.md`, written before this
  session's plot additions): grad_norm and relative_update were what let that analysis confirm
  *which* layers a freeze event hit hardest via three independent signals agreeing, and
  relative_update specifically surfaced a "freeze-shock" transient in weight-space that Fisher-drift/
  magnitude's chunk-averaging artifact made ambiguous on its own.
