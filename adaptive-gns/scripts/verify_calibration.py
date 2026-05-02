#!/usr/bin/env python3
"""
Week 2, Step 6: Verify calibration of the variance head.
- Compute ECE on validation set
- Plot sigma_i spatially on a few rollout snapshots
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns import learned_simulator, data_loader, reading_utils, noise_utils

INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3
NUM_PARTICLE_TYPES = 9


def get_simulator(data_path, model_path, model_file, device):
    """Load simulator from checkpoint."""
    from gns.train import _get_simulator
    metadata = reading_utils.read_metadata(data_path, 'train')
    simulator = _get_simulator(metadata, acc_noise_std=0.0, vel_noise_std=0.0, device=device)
    simulator.load(os.path.join(model_path, model_file))
    simulator.to(device)
    simulator.eval()
    return simulator


def compute_ece(simulator, data_path, device, dim, n_bins=10, max_batches=100, noise_std=0.0):
    """
    Compute Expected Calibration Error on validation set.
    Percentile binning (not classification ECE): bins particles by predicted sigma^2,
    checks if empirical SE matches dim*sigma^2 (for dim-D isotropic Gaussian, E[||e||^2] = dim*sigma^2).
    """
    loader = data_loader.get_data_loader_by_samples(
        path=os.path.join(data_path, 'valid.npz'),
        input_length_sequence=INPUT_SEQUENCE_LENGTH,
        batch_size=2,
        shuffle=False,
    )
    n_features = len(loader.dataset._data[0])

    all_se = []
    all_sigma_sq = []
    all_non_kinematic = []

    predict_fn = simulator.module.predict_accelerations if hasattr(simulator, 'module') else simulator.predict_accelerations

    with torch.no_grad():
        for batch_idx, example in enumerate(loader):
            if batch_idx >= max_batches:
                break
            position = example[0][0].to(device)
            particle_type = example[0][1].to(device)
            if n_features == 3:
                material_property = example[0][2].to(device)
                n_particles_per_example = example[0][3].to(device)
            else:
                material_property = None
                n_particles_per_example = example[0][2].to(device)
            labels = example[1].to(device)

            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                position, noise_std_last_step=noise_std).to(device)
            non_kinematic = (particle_type != KINEMATIC_PARTICLE_ID).bool()
            sampled_noise *= non_kinematic.view(-1, 1, 1).float()

            pred_acc, pred_variance, target_acc = predict_fn(
                next_positions=labels,
                position_sequence_noise=sampled_noise,
                position_sequence=position,
                nparticles_per_example=n_particles_per_example,
                particle_types=particle_type,
                material_property=material_property,
            )

            se = ((pred_acc - target_acc) ** 2).sum(dim=-1)  # (n,)
            sigma_sq = pred_variance ** 2  # (n,)

            all_se.append(se.cpu().numpy())
            all_sigma_sq.append(sigma_sq.cpu().numpy())
            all_non_kinematic.append(non_kinematic.cpu().numpy())

    all_se = np.concatenate(all_se)
    all_sigma_sq = np.concatenate(all_sigma_sq)
    all_non_kinematic = np.concatenate(all_non_kinematic)

    # Mask kinematic
    se = all_se[all_non_kinematic]
    sigma_sq = all_sigma_sq[all_non_kinematic]

    # Bin by sigma^2 percentiles (standard for regression; not classification ECE)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(sigma_sq, percentiles)
    bin_edges[-1] += 1e-9  # include max

    ece = 0.0
    bin_info = []
    for i in range(n_bins):
        mask = (sigma_sq >= bin_edges[i]) & (sigma_sq < bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        mean_se = se[mask].mean()
        mean_sigma_sq = sigma_sq[mask].mean()
        # For dim-D isotropic Gaussian: E[SE] = dim * sigma^2
        calibration_error = abs(mean_se - float(dim) * mean_sigma_sq)
        ece += (n_bin / len(se)) * calibration_error
        bin_info.append((n_bin, mean_se, mean_sigma_sq, calibration_error))

    return ece, bin_info


def rollout_with_variance(simulator, positions, particle_type, material_property,
                          n_particles_per_example, nsteps, device):
    """Rollout and collect (positions, sigma) at each step."""
    predict_fn = (simulator.module.predict_positions_with_variance
                  if hasattr(simulator, 'module') else simulator.predict_positions_with_variance)
    kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).bool()

    initial = positions[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth = positions[:, INPUT_SEQUENCE_LENGTH:]
    current = initial
    pred_positions = []
    variances = []

    with torch.no_grad():
        for step in range(nsteps):
            next_pos, var = predict_fn(
                current,
                nparticles_per_example=[n_particles_per_example],
                particle_types=particle_type,
                material_property=material_property,
            )
            gt_step = ground_truth[:, step]
            next_pos = torch.where(
                kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
                gt_step, next_pos,
            )
            pred_positions.append(next_pos.cpu().numpy())
            variances.append(var.cpu().numpy())
            current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    return np.stack(pred_positions), np.stack(variances), ground_truth.cpu().numpy()


def plot_sigma_spatial(positions, sigma, output_path, step_idx, bounds=None):
    """Plot particle positions coloured by predicted variance. 2D only."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping spatial plot")
        return

    pos = positions  # (n_particles, 2 or 3)
    sig = sigma      # (n_particles,)
    dim = pos.shape[-1]
    if dim != 2:
        print("Spatial plot supports 2D only; skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=sig, s=3, cmap='viridis')
    plt.colorbar(sc, ax=ax, label=r'Variance ($\sigma^2$)')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    if bounds is not None:
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    plt.tight_layout()

    # Save as PDF if the output path ends with .pdf, otherwise PNG
    out = output_path
    if not out.endswith('.pdf'):
        out = output_path.replace('.png', '.pdf')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved spatial plot to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Dataset dir (e.g. ../WaterDropSample/dataset/)')
    parser.add_argument('--model_path', required=True, help='Model dir')
    parser.add_argument('--model_file', default='latest', help='Model checkpoint')
    parser.add_argument('--output_dir', default=None, help='Output dir for plots (default: model_path)')
    parser.add_argument('--n_bins', type=int, default=10, help='ECE bins')
    parser.add_argument('--max_batches', type=int, default=100, help='Max validation batches for ECE')
    parser.add_argument('--n_rollout_trajectories', type=int, default=3, help='Trajectories for spatial plots')
    parser.add_argument('--n_rollout_steps', type=int, default=50, help='Steps per rollout')
    parser.add_argument('--plot_steps', nargs='+', type=int, default=[1, 10, 30], help='Steps to plot sigma')
    args = parser.parse_args()

    data_path = args.data_path.rstrip('/') + '/'
    model_path = args.model_path.rstrip('/') + '/'
    output_dir = args.output_dir or model_path
    os.makedirs(output_dir, exist_ok=True)

    if args.model_file == 'latest':
        candidates = glob.glob(os.path.join(model_path, 'model-*.pt'))
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints in {model_path}")
        model_file = max(candidates, key=lambda p: int(p.split('-')[-1].split('.')[0]))
        model_file = os.path.basename(model_file)
    else:
        model_file = args.model_file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = get_simulator(data_path, model_path, model_file, device)
    metadata = reading_utils.read_metadata(data_path, 'rollout')
    bounds = metadata.get('bounds', None)
    dim = metadata.get('dim', 2)  # WaterDrop/Sand are 2D

    # ECE
    print("Computing ECE...")
    ece, bin_info = compute_ece(simulator, data_path, device, dim=dim,
                                n_bins=args.n_bins, max_batches=args.max_batches)
    print(f"ECE = {ece:.6f}")
    ece_path = os.path.join(output_dir, f'ece_{model_file.replace(".pt", "")}.txt')
    with open(ece_path, 'w') as f:
        f.write(f"ECE = {ece:.6f} (dim={dim})\n")
        f.write(f"Bin: n, mean_SE, mean_sigma_sq, |mean_SE - {dim}*mean_sigma_sq|\n")
        for i, (n, me, ms, ce) in enumerate(bin_info):
            f.write(f"  {i}: {n}, {me:.6f}, {ms:.6f}, {ce:.6f}\n")
    print(f"Saved ECE to {ece_path}")

    # Calibration curve: mean SE vs dim*sigma^2 per bin
    if bin_info:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            mean_ses = [b[1] for b in bin_info]
            mean_sigma_sqs = [b[2] for b in bin_info]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(mean_sigma_sqs, mean_ses, label='bins')
            max_val = max(max(mean_sigma_sqs), max(mean_ses) / dim) if dim > 0 else max(mean_sigma_sqs)
            ax.plot([0, max_val], [0, dim * max_val], 'r--', label='perfect calibration')
            ax.set_xlabel(r'mean $\sigma^2$')
            ax.set_ylabel('mean SE')
            ax.set_title(f'Calibration curve (dim={dim})')
            ax.legend()
            ax.set_aspect('equal')
            plt.tight_layout()
            cal_path = os.path.join(output_dir, f'calibration_curve_{model_file.replace(".pt", "")}.png')
            plt.savefig(cal_path, dpi=150)
            plt.close()
            print(f"Saved calibration curve to {cal_path}")
        except ImportError:
            pass

    # Spatial plots (use trajectory loader for correct format: n_particles, T, dim)
    valid_npz = os.path.join(data_path, 'valid.npz')
    if not os.path.isfile(valid_npz):
        valid_npz = os.path.join(data_path, 'test.npz')
    if not os.path.isfile(valid_npz):
        print("No valid.npz or test.npz found, skipping spatial plots")
    else:
        ds = data_loader.get_data_loader_by_trajectories(path=valid_npz)
        # TrajectoriesDataset: (positions, particle_type, material_property?, n_particles)
        has_material = len(ds.dataset._data[0]) >= 3
        for traj_idx, features in enumerate(ds):
            if traj_idx >= args.n_rollout_trajectories:
                break
            positions = features[0].to(device)
            particle_type = features[1].to(device)
            material_property = features[2].to(device) if has_material else None
            n_particles = int(features[3]) if has_material else int(features[2])

            pred_pos, variances, _ = rollout_with_variance(
                simulator, positions, particle_type, material_property,
                n_particles, args.n_rollout_steps, device)

            for step in args.plot_steps:
                if step >= pred_pos.shape[0]:
                    continue
                out_path = os.path.join(output_dir, f'sigma_traj{traj_idx}_step{step}.pdf')
                plot_sigma_spatial(
                    pred_pos[step], variances[step],
                    out_path, step, bounds=bounds)

    print("Calibration verification complete.")


if __name__ == '__main__':
    main()
