#!/bin/bash
#SBATCH --job-name=calib_wd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/calib_wd_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/calib_wd_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python
DATA=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/dataset/
MODEL=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/adaptive_gns/
OUT=/gpfs/radev/home/az474/AdaptiveGNS/WaterDrop/models/adaptive_gns/calibration/

export PYTHONPATH=/gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns

mkdir -p "$OUT"

echo "=== WaterDrop calibration (AdaptGNS, latest checkpoint) ==="
$PYTHON /gpfs/radev/home/az474/AdaptiveGNS/adaptive-gns/scripts/verify_calibration.py \
  --data_path=$DATA \
  --model_path=$MODEL \
  --model_file=latest \
  --output_dir=$OUT \
  --n_bins=10 \
  --max_batches=100 \
  --n_rollout_trajectories=3 \
  --n_rollout_steps=100 \
  --plot_steps 1 10 50

echo "Done."
