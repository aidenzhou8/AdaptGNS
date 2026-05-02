#!/bin/bash
#SBATCH --job-name=gns_mlp
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/mlp_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/mlp_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
GNS_DIR=/gpfs/radev/home/az474/AdaptiveGNS/gns-main
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
MODELS=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/baseline_mlp/

export PYTHONPATH=$GNS_DIR:$PYTHONPATH

# nmessage_passing_steps=0 disables all message passing -> pure MLP baseline
$PYTHON -m gns.train \
  --data_path=$DATA \
  --model_path=$MODELS \
  --ntraining_steps=500000 \
  --nsave_steps=5000 \
  --lr_decay_steps=500000 \
  --nmessage_passing_steps=0 \
  --mode=train
