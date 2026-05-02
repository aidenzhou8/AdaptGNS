#!/bin/bash
#SBATCH --job-name=eval_ada
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_adaptive_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_adaptive_%j.err

# Parameters passed via sbatch --export (with defaults)
SIGMA_PCT=${SIGMA_PCT:-80}
RADIUS=${RADIUS:-1.267}

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
MODEL=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/adaptive_gns/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

echo "=== sigma_pct=${SIGMA_PCT}, radius_factor=${RADIUS} ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$DATA --model_path=$MODEL --model_file=latest \
  --sigma_percentile=${SIGMA_PCT} --radius_factor=${RADIUS}

echo "Done."
