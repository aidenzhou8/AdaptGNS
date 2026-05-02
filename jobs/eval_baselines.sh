#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/gns-main

echo "=== MLP ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/gns-main/scripts/evaluate_rollout_mse.py \
  --data_path=$DATA \
  --model_path=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/baseline_mlp/ \
  --model_file=latest \
  --nmessage_passing_steps=0

echo "=== k=5 ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/gns-main/scripts/evaluate_rollout_mse.py \
  --data_path=$DATA \
  --model_path=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/baseline_k5/ \
  --model_file=latest

echo "=== k=10 ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/gns-main/scripts/evaluate_rollout_mse.py \
  --data_path=$DATA \
  --model_path=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/baseline_k10/ \
  --model_file=latest \
  --connectivity_radius=0.019
