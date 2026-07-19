#!/bin/bash
# ViT-small, no-freeze baseline, cluster config (40 epochs).
# Submit from the repo root: sbatch slurm/train_vit_nofreeze.sh
# See slurm/README.md before your first submission -- at minimum you must
# edit --account below.

#SBATCH --account=rrg-msh
#SBATCH --job-name=vit_nofreeze
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --mail-user=bao.lugherini@student-cs.fr
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 python/3.11 scipy-stack/2024a cuda/12.2

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python src/train.py \
  --model vit_small \
  --config src/configs/config_vit_small_cluster.yaml \
  --disable-freeze
