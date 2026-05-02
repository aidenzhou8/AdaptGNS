# Paper figures

Targets used by `CPSC_4830_Final_Report.tex`.

| Figure | File | How to (re)generate |
| --- | --- | --- |
| 1. Interaction graph snapshots at Step 100 | `figures/fig_graph_snapshot.pdf` | `python adaptive-gns/scripts/make_graph_snapshot.py --waterdrop_path WaterDrop/dataset/test.npz --sand_path Sand/dataset/test.npz --step 100 --radius 0.015 --traj 0 --output figures/fig_graph_snapshot.pdf` |
| 2. Per-particle uncertainty strip | `sigma_traj0_strip_1_10_50.pdf` | `python adaptive-gns/scripts/verify_calibration.py ...` to produce per-step PNGs, then `python adaptive-gns/scripts/make_sigma_strip.py --traj 0 --steps 1 10 50` to assemble the strip. |
| 3. Per-step dynamics (edges + MSE) | `edges_and_mse_sand_waterdrop1.png` | `python adaptive-gns/scripts/plot_mse_vs_steps_sand_waterdrop.py` (consumes the rollout `.npz` files under `Sand/models/` and `WaterDrop/models/`). |

`make_graph_snapshot.py` requires the WaterDrop test set in `.npz` form. If
you only have the upstream TFRecords, run
`python adaptive-gns/utils/convert_tfrecord_to_npz.py --input_dir WaterDrop/dataset --splits test --ndim 2`
first.
