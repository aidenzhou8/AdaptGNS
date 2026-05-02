#!/bin/bash
#SBATCH --job-name=wd_k10
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/wd_k10_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/wd_k10_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
GNS_DIR=/gpfs/radev/home/az474/AdaptiveGNS/gns-main
DATA=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/dataset/
MODELS=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/baseline_k10/

export PYTHONPATH=$GNS_DIR:$PYTHONPATH
mkdir -p $MODELS

$PYTHON -m gns.train \
  --data_path=$DATA \
  --model_path=$MODELS \
  --ntraining_steps=500000 \
  --nsave_steps=5000 \
  --lr_decay_steps=500000 \
  --connectivity_radius=0.019 \
  --mode=train
