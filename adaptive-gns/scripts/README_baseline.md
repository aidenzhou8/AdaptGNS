# Week 1, Step 2: Baseline Verification

## Training time estimate

Based on Sand training (1000 steps in ~5.3 min on CPU):

| Steps   | CPU (est.) | GPU (est., ~20× faster) |
|---------|------------|-------------------------|
| 50k     | ~4.4 hr    | ~13 min                 |
| 500k    | ~44 hr     | ~2.2 hr                 |
| 2M      | ~7.4 days  | ~9 hr                   |

*Use `--ntraining_steps` and `--nsave_steps` accordingly. GPU strongly recommended for full training.*

## Hyperparameter clarification

The baselines are controlled by **connectivity radius** (not literal k-NN). Radius serves as a proxy for average degree:
- **Fixed-radius (k≈5) baseline**: `default_connectivity_radius=0.015` from metadata (~5–6 neighbors on average)
- **k≈10 baseline**: Increase radius (e.g. ~0.021) or switch to k-NN in code

## Sand baseline (fixed-radius)

### Training
```bash
cd gns-main
python -m gns.train --data_path="../Sand/dataset/" --model_path="../Sand/models/" \
  --ntraining_steps=2000000 --nsave_steps=10000 --mode=train
```
*Note: Full convergence typically needs ~2M steps. Use fewer for quick checks.*

### Evaluation
```bash
python scripts/evaluate_rollout_mse.py \
  --data_path="../Sand/dataset/" \
  --model_path="../Sand/models/" \
  --model_file="latest"
```

### Sample output (model-500.pt, 5 trajectories — undertrained)
| Step | MSE (mean) | MSE (std) |
|------|------------|-----------|
| 1    | 3.72e-08   | 4.89e-08  |
| 10   | 1.61e-04   | 1.96e-04  |
| 50   | 4.68e-02   | 3.98e-02  |
| 200  | 9.38e-01   | 1.16e+00  |

*Converged Sand baseline (from paper): rollout MSE ~2.37e-3. Run longer training for production numbers.*

### Per-step MSE output (for graphing)

The script saves `rollout_mse_<model>.npz` and `.json` in the model directory. To plot:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('Sand/models/rollout_mse_model-500.npz')
steps = data['steps']
mse_mean = data['mse_mean']
mse_std = data['mse_std']

plt.figure()
plt.semilogy(steps, mse_mean, 'b-', label='MSE (mean)')
plt.fill_between(steps, mse_mean - mse_std, mse_mean + mse_std, alpha=0.3)
plt.xlabel('Rollout step')
plt.ylabel('MSE')
plt.legend()
plt.savefig('rollout_mse.pdf')
```
