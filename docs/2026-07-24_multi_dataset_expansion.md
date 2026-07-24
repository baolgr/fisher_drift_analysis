# CIFAR-100 and Downsampled ImageNet (32x32) support added (2026-07-24)

User asked to adapt the ResNet50/ViT-small benchmark to also run on CIFAR-100 and ImageNet,
reusing the existing files/architecture rather than writing new ones, plus proper data-loading
files for the new datasets, and to reorganize `runs/` into `<Dataset>/<Model>/` subfolders. No
dataset was downloaded — the user stages data on the server.

## Key decision: Downsampled ImageNet, 32x32

Clarified with the user before implementing: "ImageNet" means the official Downsampled ImageNet
distribution (Chrabaszcz et al.), 32x32 variant, pickle-batch format — not full-resolution
224x224 ImageNet-1k. This means all three datasets share the exact same resolution as the
existing CIFAR pipeline, so **`src/models/resnet_cifar.py` and `src/models/vit_cifar.py` needed
zero architectural changes** — the CIFAR stem (no maxpool, 3x3 stride-1) applies unchanged, and
`num_classes` was already a constructor parameter on both. A full-resolution path would have
required a second ResNet stem (7x7 stride-2 + maxpool) and a larger ViT patch size — explicitly
rejected as out of scope.

## What got built

**Data loading**, split from the previous CIFAR-10-only `src/data/dataset.py` into:
- `src/data/common.py` — shared `resolve_data_root` + `seeded_train_val_split`.
- `src/data/cifar10.py` / `cifar100.py` — thin torchvision wrappers, same seeded 45k/5k-style
  split pattern, own mean/std constants.
- `src/data/imagenet32.py` — the one genuinely new loader. Parses the official
  `train_data_batch_1..N` + `val_data` pickle files (`encoding="latin1"`, channel-planar
  `(N,3072)` uint8 reshaped to `(N,3,32,32)`, 1-indexed labels). `val_data` is used as the
  held-out test set, mirroring the existing "official test set touched exactly once" convention;
  the train/val split is instead carved out of the pooled train batches, with no CIFAR-sized
  default (`val_size` must be set explicitly in the config — the pool is ~25x larger).
- `src/data/dataset.py` — now a `DatasetSpec` registry + dispatcher (`build_dataloaders` /
  `num_classes` / `display_name` per dataset key), so a future dataset is "write one file, add
  one registry entry" (raised as an explicit requirement mid-review — see below).

**`src/train.py`**: `_MODEL_REGISTRY` got the same treatment (`ModelSpec`:
`build`/`layer_registry`/`display_name`), for the same reason. `build_model`'s `num_classes` now
resolves from `cfg["num_classes"]`, falling back to the dataset registry's default. `run_training`
now nests `output_dir` as `runs/<Dataset>/<Model>/<run_name>/` using both registries'
`display_name`s.

**Configs**: `dataset`/`num_classes` added to the existing `config_resnet50.yaml`/
`config_vit_small.yaml` (defaults, no behavior change), plus 4 new base + 4 new cluster configs
for CIFAR-100/ImageNet32. The CIFAR-100 pair reuses the CIFAR-10-validated Fisher/freeze
hyperparameters unchanged (same step-count math — CIFAR-100 has the same 50k/10k split shape,
just more classes) but is flagged unverified for the harder 100-way task. The ImageNet32 pair is
flagged more strongly unverified: `num_epochs`/`freeze_interval` are copied from the CIFAR
configs and are almost certainly wrong given ImageNet32's ~25x larger per-epoch step count — must
be recomputed once the dataset is staged and real `steps_per_epoch` is known.

**SLURM**: 4 new no-freeze-only baseline jobs
(`slurm/train_{resnet,vit}_{cifar100,imagenet32}_nofreeze.sh`), per direct instruction — matches
how CIFAR-10 itself was validated (baseline first, freeze-tuning as a separate follow-up). The two
ImageNet32 jobs' `--time` budgets are rough placeholders, explicitly flagged in-file as unmeasured.

**`runs/` restructuring**: nested `runs/<Dataset>/<Model>/<run_name>/`, was flat
`runs/<model>_<freeze|nofreeze>/`. The 16 pre-existing CIFAR-10 runs were migrated into
`runs/CIFAR-10/<Model>/` via a plain filesystem `mv` — `runs/` turned out to be entirely
gitignored already (confirmed via `git check-ignore`), so this wasn't a `git mv` and no git
history was at stake. `scripts/compare_runs.py` updated to glob recursively
(`runs/**/summary.json`, was `runs/*/summary.json`) and label rows by path relative to `runs/`
rather than leaf dir name alone, so results stay unambiguous across datasets/models.
`scripts/regenerate_plots.py` needed no code change (takes explicit paths, rebuilds from the
`cfg` stored in `summary.json`) — just its usage docstring updated. `slurm/validate_setup.sh` and
`slurm/README.md`'s path references updated to match (mechanical, not a rewrite).

## Extensibility requirement (added mid-review)

The user asked, after the first plan draft, for the design to stay easy to extend to further
datasets/models later. Both registries (`DATASET_REGISTRY` in `src/data/dataset.py`,
`_MODEL_REGISTRY` in `src/train.py`) hold a small frozen dataclass (builder + metadata) instead of
a bare tuple, specifically so adding a future dataset/model touches exactly one file + one
registry line — no parallel display-name dicts or scattered special-casing to keep in sync.
Flagged explicitly in `CLAUDE.md`'s "Code style" section as the one deliberate exception to
"no abstractions beyond what's needed."

## Testing

Fast suite 60→64 (`tests/test_data_imagenet32.py`, new — tiny synthetic pickle batches built
in-test, no download, no real dataset; exercises the byte-layout reshape and 1-indexed→0-indexed
label conversion, the two places most likely to have an off-by-one bug). No new tests for
`cifar100.py` (thin torchvision wrapper, mirrors the existing untested precedent for `cifar10.py`).
`tests/test_resnet.py`/`test_vit.py`'s `test_train_smoke` had a hardcoded flat `run_dir` path that
broke under the new nesting — fixed to match (`tmp_path / "CIFAR-10" / "ResNet50" / ...`); both
slow variants re-verified end to end with a real trainer. Manually confirmed the fail-fast
contract holds for the two new datasets (unknown dataset key, missing `data_root` for both
CIFAR-100 and ImageNet32 all raise immediately with a clear message, not a hang or opaque
traceback).

## Not done (explicit scope decisions)

- `_sweep` YAML variants for CIFAR-100/ImageNet32 — only base + cluster configs.
- Freeze-enabled or sweep SLURM jobs for the new datasets — no-freeze baselines only, per the user.
- Full-resolution (224x224) ImageNet — rejected per the user's clarifying answer.
- Any real dataset download or staging — the user handles this on the server.
- `docs/plan.md` (the French rationale doc) left untouched.
