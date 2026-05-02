#!/bin/bash
#SBATCH --job-name=adapt_aug
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/adapt_aug_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/adapt_aug_%j.err

# AdaptGNS with graph augmentation during training.
# At each training step, ~30% of particles are randomly given expanded
# connectivity (radius * 1.267 = matches k=10 radius). This teaches the
# model to handle mixed-radius graphs and is meant to close the
# train/inference covariate-shift gap introduced by the inference-time
# adaptive rollout.
#
# Loss is still heteroscedastic NLL (default whenever the model returns
# a variance), so the variance head trains as before.

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
GNS_DIR=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
MODELS=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/adaptive_gns_aug/

export PYTHONPATH=$GNS_DIR:$PYTHONPATH

mkdir -p $MODELS

$PYTHON -m gns.train \
  --data_path=$DATA \
  --model_path=$MODELS \
  --ntraining_steps=500000 \
  --nsave_steps=5000 \
  --lr_decay_steps=500000 \
  --augment_radius_prob=0.30 \
  --augment_radius_factor=1.267 \
  --mode=train
