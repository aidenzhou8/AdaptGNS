#!/bin/bash
#SBATCH --job-name=wd_convert
#SBATCH --partition=day
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/convert_waterdrop_%j.out
#SBATCH --error=/gpfs/radev/home/az474/AdaptiveGNS/jobs/logs/convert_waterdrop_%j.err

PYTHON=/gpfs/radev/home/az474/.conda/envs/gns/bin/python

$PYTHON ~/AdaptiveGNS/gns-main/utils/convert_tfrecord_to_npz.py \
  --input_dir  ~/AdaptiveGNS/WaterDrop/dataset \
  --output_dir ~/AdaptiveGNS/WaterDrop/dataset \
  --splits train valid test \
  --ndim 2
