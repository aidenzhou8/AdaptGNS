#!/bin/bash
#SBATCH --job-name=eval_aug
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_aug_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_aug_%j.err

# Evaluate the augmentation-trained AdaptGNS model on Sand.
# Runs both the fixed-graph baseline (for direct compare against the
# non-augmented model) and the adaptive rollout at p70 r1.267
# (matching our reported headline configuration).

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
MODEL=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/adaptive_gns_aug/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

echo "=== AdaptGNS (aug) fixed-graph ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_rollout_mse.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest

echo "=== AdaptGNS (aug) adaptive p70 r1.267 ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest \
  --sigma_percentile=70 \
  --radius_factor=1.267

echo "=== AdaptGNS (aug) adaptive p70 r1.5 (more aggressive — only safe with augmentation) ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest \
  --sigma_percentile=70 \
  --radius_factor=1.5

echo "Done."
