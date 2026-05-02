#!/bin/bash
#SBATCH --job-name=gns_kmid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/kmid_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/kmid_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
GNS_DIR=/gpfs/radev/home/az474/AdaptiveGNS/gns-main
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
MODELS=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/baseline_kmid/

export PYTHONPATH=$GNS_DIR:$PYTHONPATH
mkdir -p $MODELS

# connectivity_radius=0.0176 gives ~22,958 edges/step, matching AdaptGNS (ours)
# ablation: does adaptive topology help, or just having more edges on average?
$PYTHON -m gns.train \
  --data_path=$DATA \
  --model_path=$MODELS \
  --ntraining_steps=500000 \
  --nsave_steps=5000 \
  --lr_decay_steps=500000 \
  --connectivity_radius=0.0176 \
  --mode=train
