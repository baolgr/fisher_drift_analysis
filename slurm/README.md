# SLURM job scripts (Alliance Canada H100 clusters — Fir or Rorqual)

Four jobs, one per (model x freeze/nofreeze): `train_{vit,resnet}_{freeze,nofreeze}.sh`. Each
runs the cluster config (`src/configs/config_{vit_small,resnet50}_cluster.yaml`, 40 epochs --
see that file's header comment for why `freeze_interval` had to be recomputed alongside
`num_epochs`, not left at the local-config value).

Verified 2026-07-19 against the actual `docs.alliancecan.ca` "Running jobs" / Fir / Rorqual /
GPU-scheduling pages (the user fetched the content directly after an automated fetch got
bot-blocked earlier the same session — two real corrections came out of that: `--gres=gpu:...`
is being phased out in favour of `--gpus-per-node=...`, and `--mem=0` was wrong for a
single-GPU-of-4 request on a shared node, see below). The module load line was additionally
verified live on Rorqual via `module spider` (two more real corrections: `python/3.11` doesn't
exist as a bare version, needs the full `python/3.11.5`; `cuda/12.6` needs an explicit compiler
prerequisite the original draft didn't have) -- see "What the scripts request" below for the
full trail. Everything else not cited to a specific page/command below is still general SLURM
knowledge, not a direct quote.

## Before your first submission

1. **`--account=def-CHANGEME`** in every script -- replace with your actual Resource Allocation
   Project (RAP) id (find it via CCDB: *Mes projets -> Mes ressources et allocations*, "Nom du
   groupe" field). Submitting without a valid `--account` fails immediately.
2. **Stage the dataset — required on Rorqual, harmless-but-still-done-here on Fir.** The two
   clusters differ: Fir's compute nodes have full internet access ("Les nœuds de calcul \[de
   Fir\] ont plein accès à l'internet"), but Rorqual's compute nodes do **not**, by policy
   ("les nœuds de calcul de Rorqual n'aient pas accès à l'internet" -- exceptions need a support
   ticket). Since you said you'll run on either, and `src/data/dataset.py` uses `download=False`
   on purpose (fails fast on a missing dataset rather than hanging on a network call), these
   scripts assume **no internet on the compute node either way** -- correct and required on
   Rorqual, and simply a no-op (data's already there, nothing to fetch) on Fir. One setup, works
   on both, rather than a cluster-specific branch. Stage `dataset/` (~340M) from your local
   machine before submitting:
   ```bash
   rsync -av dataset/ your-cluster:~/path/to/AtlasAnalyticsLab/dataset/
   ```
   Use the **login/copy node** for this (`rsync`/`scp`), not Globus-only transfer endpoints --
   both clusters' docs distinguish a Globus collection address from the plain login-node address
   for `rsync`/`scp`.
3. **Get the repo itself onto the cluster** -- `git clone`/`git push` once `git init` has happened
   locally (see CLAUDE.md: repo root isn't a git repo yet, pending confirmation), or `rsync` the
   whole tree (respecting `.gitignore` -- `rsync -av --exclude-from=.gitignore .` works even
   without git).
4. `slurm/logs/` must exist before submission (`--output` writes there) -- it's already in this
   repo via `slurm/logs/.gitkeep`, just don't delete the directory.

## Submitting

From the repo root on the cluster (paths in the scripts are relative to `$SLURM_SUBMIT_DIR`,
i.e. wherever you run `sbatch` from -- confirmed: Slurm's default `--output` also lands in "le
répertoire à partir duquel la tâche a été soumise"):

```bash
sbatch slurm/train_vit_nofreeze.sh
sbatch slurm/train_vit_freeze.sh
sbatch slurm/train_resnet_nofreeze.sh
sbatch slurm/train_resnet_freeze.sh
```

All four are independent (no dependency chain) and can run concurrently if your allocation has
the room. Track them with `sq` (your jobs only) or `squeue`; do **not** poll either in a tight
loop from a script -- the docs explicitly warn this can degrade the scheduler for everyone.
Each writes SLURM's own stdout/stderr to `slurm/logs/<jobname>-<jobid>.out`, and `train.py`
writes its own run to `runs/<model>_<freeze|nofreeze>/` as usual (summary.json, checkpoints/,
metrics_plots/*.png -- now including the 3 new Appendix-B.1-style plots, see
`docs/2026-07-19_appendix_b1_plots.md`).

## What the scripts request, and why (with sources)

- **`--gpus-per-node=h100:1`, single GPU.** The docs explicitly deprecate the `--gres=gpu:...`
  form I originally wrote here: *"il est possible que ce format ne soit plus pris en charge.
  Nous vous recommandons de le remplacer par --gpus-per-node"* -- fixed. `h100` is the correct
  short identifier on **both** Fir and Rorqual (confirmed in the GPU-availability table on the
  GPU-scheduling page), so one script works on either without editing. Both clusters' GPU nodes
  have 4x H100-80GB per node; requesting `:1` takes one of the four. Neither model needs more --
  ViT-small is ~2.7M params, ResNet50 ~25M, both on 32x32 CIFAR-10, and `train.py` has no
  `DistributedDataParallel`/multi-GPU code path to use a second GPU even if requested.
- **`--mem=32G`, not `--mem=0`.** This was a real mistake in the first draft: `--mem=0` means
  "all memory on the node," which is fine for a full-node job but inappropriate here -- a
  single-GPU-of-4 request that also grabbed all node memory would starve the other 3 GPUs of
  usable memory on a shared node, wasting resources other jobs need. 32G is generous headroom
  for these model/batch sizes (well under either cluster's per-node total: Fir GPU nodes have
  1125G/4 GPUs, Rorqual 498G/4 GPUs).
- **`--cpus-per-task=8`.** Comfortably under both clusters' documented per-GPU maximums (Fir: up
  to 12 cores/GPU; Rorqual: up to 16 cores/GPU, from each cluster's hardware page) and matches
  the `num_workers=6` set in the cluster configs with a little headroom for the main process.
  Feel free to raise it (and `num_workers` in the YAML config to match) if a job is data-loading
  bound rather than GPU bound -- check with `sacct -j <jobid> --format=Elapsed,MaxRSS` after the
  first run rather than guessing further.
- **`--time=02:00:00` (ViT) / `03:00:00` (ResNet50)**: comfortably over the documented minimum
  ("Chaque tâche devrait être d'une durée d'au moins une heure") and well under the 7-day max.
  Still an unmeasured guess for the *upper* bound though -- H100 should be faster than the local
  MPS-backend runs (~35-40 min for 18 epochs) despite these tiny models likely being
  data-loading/Python-loop-bound rather than compute-bound at this scale, but 40 epochs is
  ~2.2x the local epoch count and this hasn't actually been benchmarked on H100. Tighten after
  the first job if you want faster queue turnaround on the next submission -- given the
  2026-07-21 deadline, err on submitting sooner with a loose time limit rather than tuning this
  first.
- **No `--partition=`**: intentional, matches the docs' own guidance ("Ne pas spécifier de
  partition" -- let the scheduler assign based on requested resources).
- **`module purge` before loading modules**: matches the docs' troubleshooting guidance
  directly ("il est donc recommandé d'ajouter au script la ligne module purge avant le
  chargement des modules... pour faire en sorte que les tâches soient soumises de manière
  uniforme").
- **`module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5`**: fully verified on Rorqual
  2026-07-19 via `module spider python/3.11.5` and `module spider cuda/12.6` (not guessed --
  the first two attempts were wrong: `python/3.11` doesn't exist as a bare version, and
  `cuda/12.2` exists but needs a compiler prerequisite that wasn't in the original draft
  either). `gcc/13.3` is one of several valid non-MPI compiler choices `cuda/12.6` accepts
  (`gcc/12.3`, `gcc/13.3`, `intel/2023.2.1`, `nvhpc/25.1`, each also available with an
  `openmpi` variant) -- picked arbitrarily since this job is single-process (no MPI code
  anywhere in this project) and only ever consumes prebuilt wheels via pip, never compiles
  anything itself, so the specific compiler shouldn't matter. `scipy-stack` was dropped from
  the original draft -- redundant with what `pip install --no-index -r requirements.txt`
  already provides (numpy/matplotlib are in `requirements.txt`). If you land on Fir or Nibi
  instead of Rorqual, these exact versions aren't re-verified there -- rerun the two `module
  spider` commands above and adjust if they differ.
- **`virtualenv --no-download` + `pip install --no-index` in `$SLURM_TMPDIR`, not conda**: this
  part is carried over from earlier web-search-confirmed research this session, not re-verified
  against the pages just fetched (which didn't cover Python/package-manager setup). Still
  standard Alliance Canada guidance as far as I know, but flagging that it's on a different
  confidence footing than the corrections above.

## After a run

Per CLAUDE.md's "Running training" section: rename `runs/<model>_<freeze|nofreeze>/` to a
version-specific name if it's worth keeping (matching the existing
`runs/vit_small_freeze_v5_pure_knobs_lambda-1.0/`-style convention) before the next run of the
same model/freeze-setting overwrites it.

Sources: `docs.alliancecan.ca/wiki/Running_jobs/fr`, `.../wiki/Fir/fr`, `.../wiki/Rorqual/fr`,
and the GPU-scheduling page (`Ordonnancement_Slurm_des_tâches_avec_GPU`), all fetched directly
2026-07-19.
