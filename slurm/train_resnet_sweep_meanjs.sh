#!/bin/bash
# ResNet50, FisherAdapTune freeze enabled, cluster config, sweep probe:
# chunk_selection_metric=mean_js (lambda unchanged at -1.0). See
# src/configs/config_resnet50_cluster_sweep_meanjs.yaml for the rationale.
# Writes to runs/resnet50_freeze_sweep_meanjs/ (run_name override).
# Submit from the repo root: sbatch slurm/train_resnet_sweep_meanjs.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=resnet_sweep_meanjs
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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
  --model resnet50 \
  --config src/configs/config_resnet50_cluster_sweep_meanjs.yaml
