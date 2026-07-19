#!/bin/bash
# One-off setup validation for an interactive salloc session -- NOT an sbatch
# script (no #SBATCH directives; run manually after salloc grants a shell).
#
# Usage, from the repo root on the login node:
#   salloc --time=0:15:0 --account=def-CHANGEME --gpus-per-node=h100:1 --cpus-per-task=8 --mem=8G
#   cd fisher_drift_analysis   # or wherever you cloned/rsynced it, if not already there
#   bash slurm/validate_setup.sh
#   exit   # release the allocation once done
#
# Checks, in order: module load, venv build against the cluster's own
# wheelhouse, CUDA visibility, the fast test suite, and one real epoch on the
# real (staged) dataset on GPU -- the last step exercises dataset staging,
# the full train/val/test loop, checkpointing, and all 8 plots (including the
# 3 new Appendix-B.1-style ones) end to end, so a pass here means the actual
# sbatch jobs are very likely to run cleanly too. Uses --disable-freeze:
# freeze_interval=7200 in the cluster config never fires within 1 epoch
# (351 steps) either way, so freeze on/off exercises identical code here --
# --disable-freeze is just the simpler baseline path.
#
# Writes to runs/vit_small_nofreeze/, same path the real
# train_vit_nofreeze.sh job writes to -- the real job's output will
# overwrite this smoke-test run, so no cleanup needed, but don't mistake
# this 1-epoch summary.json for a real result if you look before rerunning.

set -euo pipefail

module purge
# Verified 2026-07-19 via 'module spider python/3.11.5' and 'module spider cuda/12.6'
# on Rorqual (both required StdEnv/2023; cuda additionally needs a compiler --
# gcc/13.3 picked arbitrarily among the non-MPI options since this job is
# single-process (no MPI code anywhere in this project) and only consumes
# prebuilt wheels via pip, never compiles anything itself -- any of the listed
# non-openmpi compiler choices should work equally.
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

echo "=== CUDA check ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.cuda.get_device_name(0))"

echo "=== fast test suite ==="
pytest tests/ -m "not slow" -q

echo "=== one real epoch, real (staged) dataset, GPU ==="
python src/train.py \
  --model vit_small \
  --config src/configs/config_vit_small_cluster.yaml \
  --disable-freeze \
  --num-epochs 1

echo "=== output check ==="
ls runs/vit_small_nofreeze/metrics_plots/
python -c "import json; print('test_accuracy:', json.load(open('runs/vit_small_nofreeze/summary.json'))['test_accuracy'])"

echo "All checks passed."
