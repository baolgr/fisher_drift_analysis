# FisherAdapTune freeze-tuning experiment log (ViT-small, CIFAR-10)

Target: `runs/vit_small_nofreeze` test accuracy **0.6543** (no freeze, 18 epochs, all 2,680,906 params trainable throughout). Goal: get the freeze-enabled run into the same order of magnitude while the freeze mechanism still does something non-trivial (real parameter reduction), without diverging from FisherAdapTune's own published method further than necessary.

Config file: `src/configs/config_vit_small.yaml`. Entrypoint: `python src/train.py --model vit_small --config src/configs/config_vit_small.yaml` (18 epochs, ~40 min/run on this machine, MPS backend). `steps_per_epoch=351`, `total_steps=6318` (45k train / batch 128, drop_last).

Each run's full stdout is saved; directories below are under `runs/` unless noted. Config snapshot (the relevant Fisher/freeze fields) is given per entry since the YAML is mutated between runs.

## TL;DR — final result

**v5** (`fisher_ema_interval=200, freeze_interval=3200, js_variance_lambda=-1.0`, no custom code) reaches **0.6502 test accuracy** (nofreeze: 0.6543, -0.4pp) while retaining **48.9%** of parameters (1,316,170 / 2,693,578). This is the config currently live in `src/configs/config_vit_small.yaml`. See the "v5" entry below for the full trajectory and per-layer analysis, and "Open threads" at the end for optional further work (not required to hit the target, listed for completeness/token-budget permitting).

```
run                                               acc    loss  trainable %params  lambda n(freeze)  m(ema)
vit_small_freeze_buggy_lambda1_interval300     0.1389  3.2566          3    0.0%     1.0       300      50
vit_small_freeze_v1_interval_fix_only          0.3687  1.7495     21,610    0.8%     0.3      2000     200
vit_small_freeze_v2_plus_numel_filter          0.5072  1.3544    195,456    7.3%     0.3      2000     200
vit_small_freeze_v3_plus_patience12            0.4937  1.4021      9,216    0.3%     0.3      2000     200
vit_small_freeze_v4_single_event_lambda03      0.5542  1.2407    156,672    5.8%     0.3      3200     200
vit_small_freeze_v5_pure_knobs_lambda-1.0      0.6502  1.0055  1,316,170   48.9%    -1.0      3200     200
vit_small_nofreeze                             0.6543  0.9767  2,680,906   99.5%     1.0       300      50
```
(regenerate with `python scripts/compare_runs.py`)

---

## v0 — original buggy config (baseline "why we're here")

- `fisher_ema_interval=50, freeze_interval=300, js_variance_lambda=1.0`, no custom filters.
- Result: **0.1389** test accuracy. Collapsed to **3 trainable params** by step 600 (epoch ~1.7).
- Saved: `runs/vit_small_freeze_buggy_lambda1_interval300/`
- Diagnosis: `fisher_ema_interval=50`/`freeze_interval=300` are the README's toy "Quick Start" smoke-test values, not its validated real-world defaults (300/3000, confirmed later in paper Appendix B.2: "optimal m and n were 300 and 3000 in all experiments"). At `freeze_interval=300` this fires ~21 freeze events over the 18-epoch run; each event recomputes `threshold = mean + λ·std` over only the *currently-trainable* chunks and keeps `score >= threshold`. At `λ=1.0` this keeps only the extreme upper tail (~7-10% empirically) *every round*, compounding geometrically to the 1-chunk floor.
- **Not promising as-is** — this is the bug, not a candidate fix.

## v1 — interval fix only (fisher_ema_interval=200, freeze_interval=2000, λ=0.3)

- Rationale: match the reference 10:1 ema:freeze accumulation ratio, space freeze events out.
- Result: **0.3686** test accuracy, early-stopped at epoch 11 (patience=6). One freeze event fired at step 2000 (epoch 5.7): 194/408 chunks retained but only **21,610 params** (0.8% of 2.68M) — chunk retention (48%) didn't translate to param retention.
- **Root cause found**: bias/LayerNorm chunks (~192 elements, sliced to ~48/chunk) systematically outscore weight-matrix chunks on `total_variation` — measured via offline categorization of a probe run's dumped scores: bias chunks kept trainable 97.4% of the time vs. 5.1% for weight-matrix chunks, at the same freeze decision. Cause: the JS-distance histogram (64 bins) is too sparse on ~48-element tensors, so per-step sampling noise reads as "still learning" more than actual weight-matrix variation does.
- **Promising direction, but incomplete** — confirmed the interval fix helps but capacity is going to the wrong (numerically tiny) place.

## v2 — + min_freeze_param_numel filter (custom code, since reverted)

- Added `min_freeze_param_numel` to `FisherAdapTuneTrainer` (excludes tensors under N elements from freeze eligibility entirely — always trainable). Set to 1024.
- Result: **0.5072** test accuracy, again early-stopped (epoch 11, patience=6) — but this time by a **freeze-shock-vs-early-stopping interaction artifact**: post-freeze val loss spike (1.28→1.81) looks like "no improvement" to a patience counter comparing against the pre-freeze best, even though loss was monotonically recovering every epoch afterward.
- Single freeze event at step 2000: 28/152 eligible chunks retained, **195,456 params** (7.3% of 2.68M) — ~15x more capacity retained than v1 for the same λ, confirming the filter fixed the intended problem.
- **Promising** — best result up to this point, but two confounds (early-stopping artifact, and the filter itself being non-original code) both needed resolving before trusting the number.

## v3 — + early_stopping_patience 6→12

- Same as v2 plus patience bump to survive one freeze-shock recovery cycle (~6 epochs observed).
- Result: **0.4937** — ran to epoch 17 (2 freeze events: step 2000 and step 4000), but *worse* than v2.
- **Key finding**: the 2nd freeze event (step 4000) cut 28→4 chunks (only 14% of what survived round 1), far more aggressive than round 1's cut at the *same* λ=0.3. Cause: `_apply_variance_freeze` recomputes mean/std over only the *surviving* population each round; after round 1 filters out the noisiest chunks, the survivors cluster tightly (std collapses much faster than mean), so the same λ prunes a much larger fraction of what's left.
- **Not promising as multi-event** — established that *any* λ, reapplied to a shrinking, self-selected population, compounds unpredictably. Motivated moving to a single-event design.

## v4 — single freeze event (freeze_interval=3200 > total_steps/2), λ=0.3, numel filter kept, patience=12

- `freeze_interval=3200` ensures only one multiple of it falls within `total_steps=6318`, so exactly one freeze decision ever fires (~step 3200, epoch ~9.1), sidestepping the v3 compounding problem entirely.
- Result: **0.5542** test accuracy, full 18 epochs, no early-stopping artifact. Single cut: 20/152 chunks, **156,672 params** (5.9% of total).
- Post-freeze accuracy recovered steadily (0.397→0.554 over 8 epochs) but **plateaued** well below the no-freeze trajectory at the same epochs (nofreeze was ~0.62-0.65 in that range) — indicates a genuine capacity ceiling, not just "needs more recovery time."
- **Promising structurally** (single-event design confirmed sound) but retained capacity (5.9%) too low for the accuracy target.

## User pushback — are min_freeze_param_numel / min_retain_fraction "real" params?

User asked whether `js_variance_lambda`, `min_freeze_param_numel`, `min_retain_fraction` are genuine FisherAdapTune parameters. Answer: only `js_variance_lambda` is (pre-existing in `FisherAdapTune/scripts/trainer.py`); the other two were written this session, not part of upstream. Given explicit preference to minimize custom code, ran a diagnostic probe to check whether a sufficiently negative `js_variance_lambda` alone (pure original knobs) could achieve comparable retained capacity to the numel filter, without needing it.

### Probe: pure knobs, λ=-1.0, no filter (compressed schedule, 4 epochs)

- Round 1 (step 350, full 408-chunk population): retained **80.6% of chunks / 46.3% of params (1,242,298)** — far more capacity than any filtered run above, on pure original knobs.
- Round 2 (step 700, same λ): chunk retention still reasonable (69%) but **params collapsed to 23,194** (98% loss) — confirms the v3 finding (round-2 dynamics betray the weight-vs-bias bias even at very permissive λ) is a property of the *iterated* rule, not of λ specifically. Reinforces: single-event design is necessary regardless of λ or filtering.
- **`min_retain_fraction` reverted** (code + config) — not needed once freeze is single-event; lowering λ directly achieves the same goal using an original parameter.

## Paper cross-reference (Appendix B.2, Figure 6) — see docs/paper_fisheradaptune.pdf

- Confirms `fisher_ema_interval=300, freeze_interval=3000` are the paper's own validated defaults ("in all experiments").
- **All ablated λ values in the paper are negative** (0 down to -1.0 for SAM2, down to -3 for SegFormer); best per model: SAM2-Tiny -0.85, SAM2-Large -0.87, SegFormer-B0/B3 -0.95. The code/config default of `+1.0` is outside the paper's entire explored range, on the wrong side of zero.
- "More conservative freezing generally improves in-distribution performance" (i.e. more negative λ → more retained → better accuracy) — direct textual support for pushing λ negative.
- Even the paper's own best-tuned results stay *close to*, not above, full fine-tuning in-distribution (e.g. SAM2-Tiny FisherAdapTune F1 63.12 vs Full-FT 64.67) — sets a realistic ceiling expectation.
- **Figure 6** (their best SAM2/SegFormer runs): "Norm All-stages" is the *first* group frozen — opposite of what v1's categorization found on our from-scratch ViT (bias/norm chunks were the *last* to be frozen). Working hypothesis: in the paper's pretrained-fine-tuning regime, norm stats are already near-converged and genuinely stabilize fast (real signal); in from-scratch training nothing is converged yet, so the small-tensor histogram-noise floor dominates and inverts the ranking. This is a *regime mismatch*, not a bug in FisherAdapTune itself — relevant context for how far a config-only fix can go.

## Subagent deep-read of remaining paper sections (pp. 1-6, 10-13, 19-20)

Full report kept in conversation history, not re-fetched here (token economy) -- key actionable points only:

- Algorithm 1's `Require:` lists **"Pre-Trained Model θ"** as input; Assumption 3.1 (slowly-varying Fisher) needs "a small learning rate from a pretrained stationary point"; Assumption 3.7 needs "a locally flat region of the pretrained loss landscape." Neither holds near random init -- textual confirmation the method's theory assumes fine-tuning, full stop.
- Algorithm 1 requires `n >= 0.02*T_max`; our `freeze_interval=3200` vs `T_max=6318` (0.02*6318=126) satisfies this comfortably. Algorithm 2 requires `m < n` (200 < 3200, satisfied).
- **Two separate EMAs confirmed present in our code as designed**: γ (paper) = `fisher_ema_decay` (smooths raw Fisher values), β (paper) = `prev_js_ema_decay` (smooths the JS-distance metric itself) -- not collapsed into one, matches Algorithm 2.
- **Possible lead**: Algorithm 2's freeze statistic `d̄_JS` is described as "a running mean over the entire accumulated history" of smoothed drift values -- this matches our `chunk_selection_metric: mean_js` option, not `total_variation` (what every run so far has used). `total_variation` is a cumulative sum of *absolute* step-to-step deltas -- mechanically, pure measurement noise (no directional signal) still adds a positive increment every step since `|Δ|>=0` regardless of sign, so it *cannot converge to zero even for a chunk whose true Fisher structure has already stabilized* the way a running mean would. Plausible mechanism for part of the small-tensor-noise bias found in v1. Worth testing `chunk_selection_metric: mean_js` as a cheap (pure existing config knob, no new code) follow-up if v5 needs further improvement.
- No mention anywhere in the paper of from-scratch applicability as even a discussed limitation/future-work item -- confirms this is genuinely unexplored territory for the method, not something we're missing a documented answer for.
- Finer-grained Figure 12 (SegFormer-B0 sweep, 12 λ values): trend confirms monotonic, `λ=-0.95` best, `λ<=-1.5` freezes almost nothing, `λ=-3.0` ~ no freezing. Our `λ=-1.0` sits just past their optimum on the conservative (retain-more) side -- reasonable given our goal is accuracy parity, not their efficiency/zero-shot-generalization trade-off.

## v5 — pure knobs, single event, λ=-1.0 (in progress / see below for outcome)

- `freeze_interval=3200` (single event, same as v4), `fisher_ema_interval=200`, `js_variance_lambda=-1.0`, **no custom trainer code** (`min_freeze_param_numel`/`min_retain_fraction` both reverted).
- Rationale: v4's numel filter got 5.9% retained capacity at λ=0.3; the pure-knobs probe above got 46.3% retained capacity at λ=-1.0 on round 1 alone (single event, no compounding) — testing whether that translates to comparable-or-better accuracy without any custom code.
- **Result: SUCCESS. Test accuracy 0.6502 vs. no-freeze 0.6543 (-0.0041, 99.4% of baseline), with 1,316,170 / 2,693,578 params retained (48.9%).** Saved: `runs/vit_small_freeze_v5_pure_knobs_lambda-1.0/`. This is the answer: pure original FisherAdapTune knobs, zero custom trainer code, reach accuracy parity with the no-freeze baseline while genuinely halving the trainable-parameter budget.
- Full trajectory: epochs 1-9 are bit-identical to the no-freeze run (deterministic, freeze hasn't fired yet). Freeze event at step 3200: 328/408 chunks (80.4%) / 1,316,170 params (49.1% of pre-freeze 2,680,906) retained -- by far the most capacity retained of any run so far (v4's filtered approach retained 5.9%). Accuracy impact of the cut itself was minimal: epoch9 0.5719 -> epoch10 0.5598 (-0.012), vs. drops of -0.15 to -0.18 in every prior freeze-enabled run. From epoch 11 onward, val accuracy oscillates *around* the no-freeze trajectory (sometimes 0.005-0.008 ahead, sometimes behind) rather than trailing it systematically -- consistent with noise, not a capacity deficit.
- **Per-layer analysis** (`runs/vit_small_freeze_v5_pure_knobs_lambda-1.0/metrics_plots/{grad_norm,fisher_drift,relative_update}.png`, free to inspect, no extra compute): freezing is fine-grained (per row-block within each tensor, not all-or-nothing per layer) and structured, not random. Reading grad-norm post-step-3200: `patch_embed.proj`, `head`, `blocks.0.mlp.fc1`, `blocks.2.attn.qkv` stay close to their pre-freeze gradient magnitude (barely affected); `blocks.0.attn.qkv` drops ~90%; `blocks.5.attn.qkv` and `blocks.5.mlp.fc2` (the late block, both attention *and* MLP) drop to near-zero. Loosely consistent with the paper's "generic-early / task-specific-late" framing turned on its head in one respect (here it's specifically the *late* block that saturates first, plausibly because a 6-block ViT's last block's representation converges fastest once the classification-relevant features are formed) -- a tentative read, not confirmed against ground truth per-chunk labels (would need an instrumented replay to verify exactly, not done -- token economy; flagging as an open thread, not a claim).
- `fisher_drift.png` shows why: chunks that already fell in `_apply_variance_freeze`'s frozen set stop appearing in `_chunk_js_history` (excluded from `collect_chunk_fisher_arrays`'s `allowed_chunks`), so a tracked layer's plotted drift after step 3200 is an average over only its *surviving* sub-chunks -- explains `blocks.5.mlp.fc2`'s visible upward jump right at step 3200 (the surviving minority were, by construction, the highest-scoring chunks all along; dropping the low-scoring majority from the average shifts it up discontinuously).
- **Why this config works where v1-v4 didn't**: (a) single freeze event (no compounding across rounds -- the dominant failure mode in v0/v3), (b) `λ=-1.0` on *pure* `total_variation` scores (no numel filter) still retains a healthy, non-token fraction because permissive-enough λ makes the small-tensor noise bias irrelevant -- at this threshold nearly everything clears the bar regardless of tensor size, so the bias never gets a chance to matter, (c) firing at step 3200 (epoch ~9, well after nofreeze's accuracy was already ~0.57) means the frozen 51% had already done most of its useful early learning.

---

## Open threads — optional, not required (target already met by v5)

- **`chunk_selection_metric: mean_js` instead of `total_variation`**: a subagent's close reading of the paper's Algorithm 2 (Appendix C) found the freeze statistic described there is "a running mean over the entire accumulated history" of drift values -- matches `mean_js`, not `total_variation` (what every run above used). `total_variation` is a cumulative sum of *absolute* deltas, which cannot converge to zero even for a chunk whose true Fisher structure has fully stabilized (pure symmetric noise still adds a positive increment every step) -- a plausible mechanism for the small-tensor noise bias found in v1. Cheap to test (one config line, no new code) if closer paper-fidelity or a tighter accuracy gap is wanted later.
- Try λ in the paper's actual validated sweet spot (-0.85 to -0.95) instead of -1.0 -- v5's -1.0 sits just past their optimum on the conservative/retain-more side (confirmed via Figure 12's finer sweep), which is why it undershoots their typical freeze aggressiveness but happens to suit an accuracy-parity goal well. Only worth revisiting if v5's 48.9% retained-param figure is considered too high for a meaningful efficiency story.
- Try moving the single event later (e.g. step ~4200-4500, epoch ~12-13) to see if a smaller retained fraction can still hold accuracy, now that timing-plus-single-event is confirmed to work.
- Exact per-chunk identity of what survived (which row-blocks, not just which layers) is inferred from grad-norm/fisher-drift plots in the v5 entry above, not verified against `trainer._trainable_chunks` directly -- would need a small instrumented replay (deterministic, same seed, stop at step 3200) if the advisor wants the precise list rather than the plot-based read.
- ResNet50: config prepared (`src/configs/config_resnet50.yaml`, mirrors v5's values) but **not run** per explicit instruction -- same reasoning should transfer (single late event, permissive negative λ) but BatchNorm's channel-count-scaling (64→512) vs. conv weight scaling (channels²) means the small-tensor disparity could be worse at deep stages; unverified until actually run.
