# ViT-small v5 per-layer curve analysis (2026-07-19)

Visual read of the 5 plots in `runs/vit_small_freeze_v5_pure_knobs_lambda-1.0/metrics_plots/`
(`fisher_drift.png`, `fisher_magnitude.png`, `grad_norm.png`, `relative_update.png`,
`all_metrics_grid.png` — the grid is the same 4 aligned side by side, not reviewed separately),
checked against the pipeline-correctness goals in `plan.md` / `docs/2026-07-18_session-notes.md`
and the 4-metric design rationale in `CLAUDE.md`. No new runs, no code changes — this is a first-hand
look at curves the training run already produced, not a re-derivation of the prose already in
`CLAUDE.md`'s "Current status" (that prose is independently confirmed below, plus one new angle).

## Pipeline-correctness checks (plan.md / session-notes.md's stated goals)

- **Metric-vs-iteration only, never metric-vs-metric**: confirmed — all 5 figures plot x=step,
  y=one metric, one line per layer. Matches the explicit requirement in `plan.md` line 6.
- **"Courbes de dérive par couche ont une forme qualitativement sensée"**: confirmed —
  `fisher_drift.png` is smooth, monotonic, saturating growth from 0 (consistent with
  `js_distance_mode="log"`), no NaNs, no flat-at-zero lines, no un-physical jaggedness outside the
  one expected freeze discontinuity. This is the qualitative-sanity check `plan.md` asks for.
- **Attention layers show real Fisher signal**: `blocks.0/2/5.attn.qkv` all track the same healthy
  rising curve as everything else in `fisher_drift.png` — none are flat at zero. This is a direct,
  visual confirmation that the `.attn.qkv` prefix-match chunk resolution (`layers.py`'s
  `resolve_layer_chunk_keys`) is working, which is the exact "subtlety on attention layers" the
  advisor was flagged as worried about in `plan.md`.
- **All 4 metrics share one x-axis grid**: confirmed — every plot ticks 0, 200, 400, ..., ~6200
  (matches `total_steps=6318`, `fisher_ema_interval=200`). Validates the single-callback-instant
  design (`CLAUDE.md`: all 4 metrics read at the same point so curves are directly comparable).

## Resolves session-notes.md's "Critical finding" / "Open question"

`docs/2026-07-18_session-notes.md` recorded the pre-retune state: trainable params collapsing
2,680,906 → **3** by step 600, accuracy 13.9%, and left an open question — accept the collapse as a
first result, or retune the freeze config. These v5 curves are the visual answer: a **single**
discontinuity at step 3200 (not a cascading series of ever-smaller freeze events), and only 2 of 7
tracked layers are hit hard — nothing resembling the old runaway collapse. The retune was the right
call, and it shows up directly in the shape of the curves, not just in the final accuracy number.

## Per-layer story, cross-validated across all 4 metrics

Three tiers emerge, and every metric agrees on the membership:

- **Tier 1 — essentially undisturbed**: `patch_embed.proj`, `head`, `blocks.0.mlp.fc1`,
  `blocks.2.attn.qkv`. No kink in `fisher_drift`/`fisher_magnitude` at step 3200; `grad_norm`
  continues in the same band before/after; `relative_update` shows at most a small bump
  (`blocks.0.mlp.fc1`/`blocks.2.attn.qkv` ~0.06→~0.09, `head` ~0.04→~0.10) — a mild echo of the
  freeze event even in layers that were mostly spared, but nothing like Tier 3.
- **Tier 2 — partially cut**: `blocks.0.attn.qkv`. `grad_norm` drops from ~0.9-1.0 right before
  step 3200 to ~0.1 right after (~90% drop, matches the figure already in `CLAUDE.md`);
  `relative_update` spikes to ~0.21 at the transition then settles near-zero.
- **Tier 3 — hit hardest**: `blocks.5.attn.qkv` and `blocks.5.mlp.fc2`. Both were *already* the
  lowest-`grad_norm` tracked layers before step 3200 even arrived (a pre-existing trend, not
  something the freeze event invented), then both collapse to near-zero `grad_norm` after it.

This is a genuine early-vs-late asymmetry — block 5 saturates/gets frozen first, blocks 0/2 stay
active — exactly the axis `layers.py`'s registry (0/2/5 spread, attn.qkv/mlp crossed at both ends)
was built to expose. It worked: the registry design choice pays off in a real, visible finding.

## Why 4 metrics, not just Fisher drift — confirmed concretely

`patch_embed.proj` is the cleanest real example of the exact ambiguity `CLAUDE.md`'s rationale
describes: its `fisher_magnitude` is the *lowest* of all 7 layers (~0.044, ~4.5x below the ~0.20
cluster everything else sits in) and stays flat, yet its `fisher_drift` is the *highest* of the
non-discontinuous group (~0.73 vs the ~0.53-0.55 cluster) and still visibly rising at step 6200 —
i.e. still-shifting curvature, but consistently weak curvature. Drift alone would flag this layer as
"most unstable"; magnitude clarifies that instability isn't backed by strong Fisher information.
`grad_norm` independently corroborates the "still highly active" read (highest and noisiest gradient
norm of all 7 layers throughout, unaffected by the freeze event) via a signal that has nothing to do
with Fisher at all — three independent metrics agreeing, which is the whole point of tracking them together.

## New observation: `relative_update.png`'s freeze-shock signature

Not previously written up in `CLAUDE.md`/`docs/fisheradaptune_freeze_experiments.md`. Every curve in
`relative_update.png` spikes at the step-3200→3400 transition, but the spike size tracks the tiers
above almost exactly: `blocks.5.mlp.fc2` peaks at **~0.40** (the single highest point in the whole
figure, ~4x its own pre-freeze steady-state level and ~4x the next-highest spike), `blocks.0.attn.qkv`
and `blocks.5.attn.qkv` peak at ~0.20-0.21, the Tier-1 layers barely move (`patch_embed.proj`:
~0.02→~0.04). Post-transient, the curves settle into two clean, separated bands: Tier 2/3 layers
drop to a near-zero band (~0.003-0.01), Tier 1 layers settle to a lower-than-before-but-still-visible
band (~0.04-0.065) — a third, independent metric reproducing the same layer split as `grad_norm` and
`fisher_magnitude`/`fisher_drift`.

This looks like the weight-space signature of "freeze-shock" — the same phenomenon the freeze-tuning
log already names for the *val-loss* dip seen in earlier freeze-enabled runs (v2's entry:
"freeze-shock-vs-early-stopping interaction artifact") — now visible per-layer instead of only in the
aggregate validation curve, and concentrated almost entirely in the layers that got cut hardest.
Checked `src/fisher/trainer.py`'s `fit()` loop: there's no optimizer-state reset at the freeze step
(`apply_chunk_masks` only updates masks, `optimizer.state` persists across the event), which rules
out an Adam-cold-restart explanation. The precise mechanism for why it manifests as an *upward* spike
rather than a plain drop-to-zero is not nailed down — flagging as an open thread, not a claim, same
epistemic bar as the rest of the freeze-tuning log. Doesn't affect the validated accuracy result
either way (`runs/.../summary.json` is unaffected by how the recorder's derived plot looks).

## One precision caveat

The plotted `fisher_drift` per layer is the *latest single* JS-distance reading
(`trainer._chunk_js_history[k][-1][1]`), averaged over a layer's currently-tracked chunks. The freeze
decision itself scores chunks on `chunk_selection_metric="total_variation"` (a cumulative sum of
absolute deltas) — related but not the identical quantity. Reading "which layer got frozen" off
`fisher_drift.png`'s shape is a good qualitative proxy (and the tiers above are confirmed by three
*other*, independent metrics, not by this one alone) but isn't a literal readout of the freeze scoring
signal. Worth being precise about this distinction if presenting these plots to the advisor.

## Bottom line

The v5 curves satisfy both documents' stated bars: `plan.md`/session-notes.md's pipeline-correctness
and qualitative-sanity checks pass, attention layers demonstrably carry real signal, and the
session-notes.md open question (accept collapse vs. retune) is visibly resolved in favor of the
retune — a single, structured, partial freeze event instead of a runaway collapse. `CLAUDE.md`'s
4-metric design is validated concretely, not just asserted: the metrics agree with each other on a
real early-vs-late finding (block 5 saturates first) and one of them (`fisher_magnitude` vs
`fisher_drift` on `patch_embed.proj`) resolves exactly the kind of ambiguity the design was built to
catch.
