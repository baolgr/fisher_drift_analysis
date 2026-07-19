#!/bin/bash
# ResNet50, FisherAdapTune freeze enabled, cluster config (40 epochs).
# Submit from the repo root: sbatch slurm/train_resnet_freeze.sh
# See slurm/README.md before your first submission.
# --account is already set to def-msh-ab -- confirmed via `sshare -U` (rrg-msh has no GPU share).
# NOT YET RUN AT ALL for ResNet50 (any epoch count) -- see src/configs/config_resnet50.yaml for
# the untested BatchNorm-freeze caveat; treat this run's freeze-vs-nofreeze gap as a first real
# result, not a re-confirmation of the ViT finding.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=resnet_freeze
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

python src/train.py \
  --model resnet50 \
  --config src/configs/config_resnet50_cluster.yaml
