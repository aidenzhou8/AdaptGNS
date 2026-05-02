#!/bin/bash
#SBATCH --job-name=eval_gen2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_gen2_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/eval_gen2_%j.err

# Generalization test: WaterDrop-trained model evaluated on Sand data.
# Opposite direction of eval_generalization.sh.

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
SAND_DATA=/gpfs/radev/home/az474/AdaptiveGNS/Sand/dataset/
WD_MODEL=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/adaptive_gns/
OUT=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/adaptive_gns/generalization/
export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

mkdir -p $OUT

echo "=== Generalization: WaterDrop model (fixed-graph) on Sand ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_rollout_mse.py \
  --data_path=$SAND_DATA \
  --model_path=$WD_MODEL \
  --model_file=model-500000.pt \
  --output $OUT/rollout_mse_wd_on_sand

echo "=== Generalization: WaterDrop model (adaptive p70 r1.267) on Sand ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/evaluate_adaptive_rollout.py \
  --data_path=$SAND_DATA \
  --model_path=$WD_MODEL \
  --model_file=model-500000.pt \
  --sigma_percentile=70 \
  --radius_factor=1.267 \
  --output $OUT/adaptive_rollout_mse_wd_on_sand_p70_r1.267
