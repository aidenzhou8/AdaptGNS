#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_gen_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_gen_%j.err

# Generalization test: Sand-trained model evaluated on WaterDrop data.
# Tests whether the adaptive graph mechanism transfers across physics types.
# No retraining — uses the Sand model weights as-is.

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
WD_DATA=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/dataset/
SAND_MODEL=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/adaptive_gns/
OUT=/gpfs/radev/home/az474/AdaptiveGNS/Sand/models/adaptive_gns/generalization/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

mkdir -p $OUT

echo "=== Generalization: Sand model (fixed-graph) on WaterDrop ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_rollout_mse.py \
  --data_path=$WD_DATA \
  --model_path=$SAND_MODEL \
  --model_file=model-500000.pt \
  --output $OUT/rollout_mse_sand_on_wd

echo "=== Generalization: Sand model (adaptive p70 r1.267) on WaterDrop ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$WD_DATA \
  --model_path=$SAND_MODEL \
  --model_file=model-500000.pt \
  --sigma_percentile=70 \
  --radius_factor=1.267 \
  --output $OUT/adaptive_rollout_mse_sand_on_wd_p70_r1.267
