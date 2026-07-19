#!/bin/bash
# ViT-small, FisherAdapTune freeze enabled, cluster config, FIX ATTEMPT #3:
# js_variance_lambda=-1.6 instead of -1.0 (total_variation unchanged), a
# second point on the lambda sweep alongside train_vit_freeze_fix_lambda13.sh.
# See src/configs/config_vit_small_cluster_fix_lambda16.yaml and
# docs/2026-07-19_vit_cluster_run_curve_analysis.md §6 for the rationale.
# Writes to runs/vit_small_freeze_fix_lambda16/ (run_name override, distinct
# from runs/vit_small_freeze/ -- safe to submit alongside the other jobs).
# Submit from the repo root: sbatch slurm/train_vit_freeze_fix_lambda16.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=vit_freeze_fix_lambda16
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
# #SBATCH --mail-user=you@example.com
# #SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge

module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python src/train.py \
  --model vit_small \
  --config src/configs/config_vit_small_cluster_fix_lambda16.yaml
