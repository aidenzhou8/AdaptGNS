#!/bin/bash
#SBATCH --job-name=eval_wd_ada
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_wd_adaptive_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_wd_adaptive_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
DATA=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/dataset/
MODEL=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/adaptive_gns/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

# Fixed-graph eval (NLL model at default radius)
echo "=== AdaptGNS fixed-graph ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_rollout_mse.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest

# Adaptive rollout (best config from Sand sweep)
echo "=== AdaptGNS adaptive pct=70 r=1.267 ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest \
  --sigma_percentile=70 \
  --radius_factor=1.267
