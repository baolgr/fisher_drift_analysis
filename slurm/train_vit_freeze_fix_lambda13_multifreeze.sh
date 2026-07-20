#!/bin/bash
# ViT-small, FisherAdapTune freeze enabled, cluster config, MULTI-FREEZE
# variant of the recommended fix: js_variance_lambda=-1.3, total_variation,
# freeze_interval lowered 7200 -> 4000 for 3 freeze events instead of 1. See
# src/configs/config_vit_small_cluster_fix_lambda13_multifreeze.yaml for the
# full rationale (progressive-freezing observation, timing/risk assessment --
# the lowest cumulative-freeze-risk config of the 4 submitted tonight).
# Based directly on slurm/train_vit_freeze_fix_lambda13.sh, config swapped only.
# Writes to runs/vit_small_freeze_fix_lambda13_multifreeze/ (does not
# overwrite the single-event run).
# Submit from the repo root: sbatch slurm/train_vit_freeze_fix_lambda13_multifreeze.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=vit_freeze_fix_lambda13_multifreeze
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
  --config src/configs/config_vit_small_cluster_fix_lambda13_multifreeze.yaml
