# AdaptGNS: Adaptive Interaction Graphs for Particle Simulation

Code release for *Adaptive Interaction Graphs for Particle Simulation* (Aiden
Zhou, Yale University; CPSC 4830 final project, Spring 2026).
See `[CPSC_4830_Final_Report.pdf](CPSC_4830_Final_Report.pdf)` for the
write-up.

**TL;DR.** A standard GNS ([Sanchez-Gonzalez et al., 2020](https://arxiv.org/abs/2002.09405))
is augmented with a per-particle variance head trained jointly under a
heteroscedastic Gaussian NLL. At inference time, particles whose predicted
variance is above the 70th percentile receive an enlarged neighbourhood
(`radius_factor = 1.267`); the rest keep the default `r = 0.015`. The
variance estimate from the previous step drives the current step's graph,
so the cost is one forward pass per step. AdaptGNS achieves a strict
Pareto improvement on WaterDrop (20% fewer edges than fixed `k=5`, lower
MSE@200) and a modest gain on Sand.

## Repository layout

```
AdaptiveGNS/
├── CPSC_4830_Final_Report.tex     # paper source
├── README.md                      # this file
├── .gitignore
├── adaptive-gns/                  # fork of geoelements/gns with VarianceHead + adaptive loop
│   ├── gns/                       # core simulator (cited in the paper)
│   │   ├── graph_network.py       # VarianceHead lives here
│   │   ├── learned_simulator.py   # predict_positions_adaptive lives here
│   │   ├── train.py               # NLL loss + adaptive rollout entry points
│   │   └── ...
│   ├── scripts/                   # evaluation, calibration, plotting
│   ├── utils/                     # dataset converters (HDF5/TFRecord -> npz)
│   ├── slurm_scripts/             # upstream SLURM templates
│   ├── test/                      # pytest suite (inherited from upstream)
│   ├── requirements.txt
│   ├── references.bib             # upstream's bib (kept for attribution)
│   └── license.md                 # upstream MIT license
├── jobs/                          # actual SLURM scripts run on Yale Misha
├── figures/                       # paper figures (regen instructions in figures/README.md)
├── calibration_plots/paper/       # final per-particle uncertainty PDFs (Figure 2 source)
├── combined_results_plots/        # Figure 3 source (edges + MSE per step)
├── Sand/
│   ├── dataset/metadata.json      # full dataset is downloaded; see "Data" below
│   └── models/                    # rollout summary .npz files (numbers behind Table 4)
└── WaterDrop/
    ├── dataset/metadata.json
    └── models/
```

## Setup

Tested on Python 3.10 with CUDA 11.8.

```bash
git clone https://github.com/<your-username>/AdaptiveGNS.git
cd AdaptiveGNS

# Recommended: create an isolated environment.
python3 -m venv .venv
source .venv/bin/activate

# torch + torch_geometric extensions need a matching wheel index.
# Replace cu118 with your CUDA version (or cpu).
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# Remaining requirements.
pip install -r adaptive-gns/requirements.txt

# Make the package importable.
export PYTHONPATH=$PWD/adaptive-gns:$PYTHONPATH
```

Optional, for the TFRecord -> npz conversion only:

```bash
pip install tensorflow-cpu
```

## Data

Datasets are not bundled in this repo (the full files exceed GitHub's per-file
limit). Both are publicly hosted.

### Sand (Taichi-MPM, from Kumar et al. 2022)

DesignSafe-CI DOI: [10.17603/ds2-0phb-dg64](https://doi.org/10.17603/ds2-0phb-dg64).
Place the resulting files at `Sand/dataset/`:

```
Sand/dataset/
├── metadata.json
├── train.npz
├── valid.npz
└── test.npz
```

### WaterDrop (SPH, from DeepMind GNS)

Available from the original GNS release. The files ship as TFRecords; convert
to `.npz` once with:

```bash
python adaptive-gns/utils/convert_tfrecord_to_npz.py \
    --input_dir  WaterDrop/dataset \
    --output_dir WaterDrop/dataset \
    --splits train valid test \
    --ndim 2
```

Final layout:

```
WaterDrop/dataset/
├── metadata.json
├── train.npz
├── valid.npz
└── test.npz
```

## Reproducing the paper

All commands below assume `PYTHONPATH=$PWD/adaptive-gns`. Training each model
takes ~6 hours on a single A100; full evaluation per dataset is ~30 min.

### 1. Exploratory data analysis (Section 2)

```bash
python adaptive-gns/scripts/dataset_eda.py \
    --data_path WaterDrop/dataset/test.npz \
    --step 100 --radius 0.015 --n_trajectories 30
python adaptive-gns/scripts/dataset_eda.py \
    --data_path Sand/dataset/test.npz \
    --step 100 --radius 0.015 --n_trajectories 30
```

This reproduces the numbers in Table 1.

### 2. Training

The actual SLURM submissions are in `jobs/`. To run locally without a
scheduler:

```bash
# AdaptGNS (NLL loss, variance head trained jointly).
python -m gns.train \
    --data_path=Sand/dataset/ \
    --model_path=Sand/models/adaptive_gns/ \
    --ntraining_steps=500000 \
    --nsave_steps=5000 \
    --lr_decay_steps=500000 \
    --mode=train

# Same recipe, swap data path for WaterDrop.
```

Baselines `k=5`, `k=10`, `k_mid` (`r=0.0176`), and the per-particle MLP
were trained with the unmodified upstream
[geoelements/gns](https://github.com/geoelements/gns) so that their loss
remained MSE rather than NLL. Clone it alongside this repo and point the
training scripts at it:

```bash
git clone https://github.com/geoelements/gns.git ../gns-upstream
PYTHONPATH=$PWD/../gns-upstream python -m gns.train \
    --data_path=Sand/dataset/ \
    --model_path=Sand/models/baseline_k5/ \
    --connectivity_radius=0.015 \
    --ntraining_steps=500000 --mode=train

# k=10: --connectivity_radius=0.019
# k_mid: --connectivity_radius=0.0176
# MLP:  --nmessage_passing_steps=0
```

The `jobs/train_baseline_*.sh` and `jobs/eval_baselines.sh` SLURM scripts
encode the same recipes for the Misha cluster (and refer to a separate
`gns-main/` clone of the upstream).

### 3. Evaluation

```bash
# Adaptive rollout with the paper's tuned hyperparameters.
python adaptive-gns/scripts/evaluate_adaptive_rollout.py \
    --data_path=Sand/dataset/ \
    --model_path=Sand/models/adaptive_gns/ \
    --model_file=latest \
    --sigma_percentile=70 \
    --radius_factor=1.267

# Standard MSE rollout (any baseline or AdaptGNS-as-fixed-graph).
python adaptive-gns/scripts/evaluate_rollout_mse.py \
    --data_path=Sand/dataset/ \
    --model_path=Sand/models/baseline_k5/ \
    --model_file=latest
```

The grid sweep behind Section 4.1 (`sigma_percentile in {65,70,75,80,90}`,
`radius_factor in {1.2,1.267,1.5,2}`) was driven by
`jobs/eval_adaptive_rollout.sh` with `--export=SIGMA_PCT=70,RADIUS=1.267`
etc.; the resulting `.npz` files are kept in
`Sand/models/adaptive_gns/adaptive_rollout_*.npz`.

### 4. Calibration analysis (Table 3)

```bash
python adaptive-gns/scripts/verify_calibration.py \
    --data_path=Sand/dataset/ \
    --model_path=Sand/models/adaptive_gns/ \
    --model_file=latest --ntrajectories=30
```

Writes per-decile expected calibration error and the per-particle
`sigma_traj*.png` snapshots used to assemble the strip in Figure 2.

### 5. Figures


| Figure                            | Script                                                                                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Graph snapshots at Step 100    | `python adaptive-gns/scripts/make_graph_snapshot.py --waterdrop_path WaterDrop/dataset/test.npz --sand_path Sand/dataset/test.npz --output figures/fig_graph_snapshot.pdf` |
| 2. Per-particle uncertainty strip | `python adaptive-gns/scripts/verify_calibration.py ...` then `python adaptive-gns/scripts/make_sigma_strip.py --traj 0 --steps 1 10 50`                                    |
| 3. Per-step edges + MSE           | `python adaptive-gns/scripts/plot_mse_vs_steps_sand_waterdrop.py`                                                                                                          |


The rollout `.npz` files committed under `Sand/models/` and `WaterDrop/models/`
are sufficient input for Figure 3 without retraining.

## What this repository does *not* contain

To keep the repo within GitHub's size budget, the following are deliberately
absent and must be obtained as documented above:

- Raw datasets (`Sand/dataset/{train,valid,test}.npz`,
`WaterDrop/dataset/{train,valid,test}.tfrecord` or `.npz`).
- Trained model checkpoints (`*.pt`). Re-train via the commands above, or
contact the author for the Misha-cluster snapshots.
- The unmodified upstream [geoelements/gns](https://github.com/geoelements/gns)
used to train the MSE baselines (clone separately).

## Citation

If you build on this work, please cite both this project and the upstream GNS:

```bibtex
@misc{zhou2026adaptgns,
  title  = {Adaptive Interaction Graphs for Particle Simulation},
  author = {Zhou, Aiden},
  year   = {2026},
  note   = {CPSC 4830, Yale University},
}

@inproceedings{sanchezgonzalez2020learning,
  title     = {Learning to Simulate Complex Physics with Graph Networks},
  author    = {Sanchez-Gonzalez, Alvaro and Godwin, Jonathan and Pfaff, Tobias
               and Ying, Rex and Leskovec, Jure and Battaglia, Peter},
  booktitle = {ICML},
  year      = {2020},
}
```

## Acknowledgments

`adaptive-gns/` is a fork of [geoelements/gns](https://github.com/geoelements/gns)
(MIT-licensed; see `[adaptive-gns/license.md](adaptive-gns/license.md)` and
`[adaptive-gns/AUTHORS.md](adaptive-gns/AUTHORS.md)`). The non-trivial
modifications relative to upstream are the variance head in
`[adaptive-gns/gns/graph_network.py](adaptive-gns/gns/graph_network.py)`,
the heteroscedastic-NLL training loop and adaptive rollout in
`[adaptive-gns/gns/learned_simulator.py](adaptive-gns/gns/learned_simulator.py)`
and `[adaptive-gns/gns/train.py](adaptive-gns/gns/train.py)`, and everything
under `[adaptive-gns/scripts/](adaptive-gns/scripts/)`.

Compute provided by the [Yale Center for Research Computing](https://research.computing.yale.edu/)
on the Misha cluster.