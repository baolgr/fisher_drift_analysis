#!/bin/bash
# ResNet50, no-freeze baseline, Downsampled ImageNet (32x32) cluster config.
# Submit from the repo root: sbatch slurm/train_resnet_imagenet32_nofreeze.sh
#
# --time is a rough placeholder, not measured: config_resnet50_imagenet32_cluster.yaml's
# num_epochs/freeze_interval are themselves unverified (~25x CIFAR's per-epoch
# step count, see that file's header) so the real wall-clock budget is unknown
# until the dataset is staged and a short trial run establishes steps/sec.
# Re-check this value (and the config's num_epochs) before relying on it.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=resnet_imagenet32_nofreeze
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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
  --config src/configs/config_resnet50_imagenet32_cluster.yaml \
  --disable-freeze
