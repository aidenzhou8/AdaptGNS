#!/usr/bin/env python3
"""
Week 3: Evaluate adaptive rollout MSE at every step.

At each rollout step the model runs a two-pass adaptive graph:
  Pass 1 (fixed radius r) → per-particle sigma_i
  Pass 2 (augmented graph: base edges + extended edges for high-sigma particles) → next position

Results are saved in the same .npz / .json format as evaluate_rollout_mse.py so
all curves can be plotted together.
"""
import os
import sys
import argparse
import json
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns import learned_simulator, data_loader, reading_utils

INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3
EVAL_STEPS = [1, 10, 50, 200]


def adaptive_rollout_mse(simulator, positions, particle_type, material_property,
                         n_particles_per_example, nsteps, device,
                         sigma_percentile, radius_factor):
    """Single-pass lagged-sigma adaptive rollout; returns per-step MSE and edge counts.

    Step 0 uses the base graph (no prior sigma).  From step 1 onward the graph
    topology is built from sigma of the *previous* step, so each step requires
    exactly ONE GNN forward pass — the same cost as the fixed-graph baselines.

    Returns:
      mse_per_step: np.ndarray (nsteps,)
      edge_counts: np.ndarray (nsteps,) — number of edges used at each step
    """
    initial_positions = positions[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth = positions[:, INPUT_SEQUENCE_LENGTH:]
    current = initial_positions

    kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).bool()
    non_kinematic = ~kinematic_mask

    predictions = []
    edge_counts = []

    # Step 0: base graph — obtains sigma_0 which drives step 1's graph
    node_features, edge_index, edge_features = simulator._encoder_preprocessor(
        current,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_type,
        material_property=material_property,
    )
    edge_counts.append(edge_index.shape[1])
    pred_acc, sigma_prev = simulator._encode_process_decode(node_features, edge_index, edge_features)
    next_pos = simulator._decoder_postprocessor(pred_acc, current)

    gt_step = ground_truth[:, 0]
    next_pos = torch.where(
        kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
        gt_step, next_pos,
    )
    predictions.append(next_pos)
    current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    for step in range(1, nsteps):
        # Build adaptive graph using lagged sigma
        tau = torch.quantile(sigma_prev, sigma_percentile / 100.0)
        high_sigma_mask = sigma_prev > tau
        most_recent_position = current[:, -1]
        r_large = simulator._connectivity_radius * radius_factor
        senders, receivers = simulator._build_adaptive_edge_index(
            most_recent_position, [n_particles_per_example], high_sigma_mask, r_large)
        edge_counts.append(senders.shape[0])

        next_pos, sigma_prev = simulator._forward_with_edge_index(
            current,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_type,
            senders=senders,
            receivers=receivers,
            material_property=material_property,
        )
        gt_step = ground_truth[:, step]
        next_pos = torch.where(
            kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
            gt_step, next_pos,
        )
        predictions.append(next_pos)
        current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    predictions = torch.stack(predictions)          # (nsteps, nparticles, dim)
    gt = ground_truth.permute(1, 0, 2)              # (nsteps, nparticles, dim)

    se = (predictions - gt) ** 2
    se_masked = se * non_kinematic[None, :, None].float()
    n_non_kinematic = non_kinematic.sum().item()
    mse_per_step = se_masked.sum(dim=(1, 2)) / (n_non_kinematic * predictions.shape[-1])

    return mse_per_step.cpu().numpy(), np.array(edge_counts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate adaptive two-pass rollout MSE')
    parser.add_argument('--data_path', required=True,
                        help='Dataset directory (contains test.npz, metadata.json)')
    parser.add_argument('--model_path', required=True,
                        help='Adaptive-GNS model directory')
    parser.add_argument('--model_file', default='latest',
                        help='Checkpoint filename or "latest"')
    parser.add_argument('--sigma_percentile', type=float, default=80.0,
                        help='Particles above this sigma percentile get expanded edges (default: 80)')
    parser.add_argument('--radius_factor', type=float, default=1.267,
                        help='Multiplier for expanded connectivity radius (default: 1.267)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Limit evaluation to N test trajectories')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path stem (default: <model_path>/adaptive_rollout_mse_<model>)')
    args = parser.parse_args()

    data_path = args.data_path.rstrip('/') + '/'
    model_path = args.model_path.rstrip('/') + '/'

    if args.model_file == 'latest':
        candidates = glob.glob(os.path.join(model_path, 'model-*.pt'))
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints in {model_path}")
        model_file = max(candidates, key=lambda p: int(p.split('-')[-1].split('.')[0]))
        model_file = os.path.basename(model_file)
    else:
        model_file = args.model_file

    output_stem = args.output
    if output_stem is None:
        model_basename = model_file.replace('.pt', '')
        output_stem = os.path.join(
            model_path,
            f'adaptive_rollout_mse_{model_basename}_p{int(args.sigma_percentile)}_r{args.radius_factor}')
    # Strip any extension that might have been passed
    for ext in ('.npz', '.json'):
        if output_stem.endswith(ext):
            output_stem = output_stem[:-len(ext)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model:  {model_file}")
    print(f"Sigma percentile: {args.sigma_percentile}  |  Radius factor: {args.radius_factor}")

    metadata = reading_utils.read_metadata(data_path, 'rollout')
    from gns.train import _get_simulator
    simulator = _get_simulator(metadata, acc_noise_std=0.0, vel_noise_std=0.0, device=device)
    simulator.load(os.path.join(model_path, model_file))
    simulator.to(device)
    simulator.eval()

    ds = data_loader.get_data_loader_by_trajectories(path=data_path + 'test.npz')
    has_material = len(ds.dataset._data[0]) == 3

    mse_at_steps = {s: [] for s in EVAL_STEPS}
    mse_all_trajectories = []
    edge_counts_all = []

    n_eval = 0
    with torch.no_grad():
        for i, features in enumerate(ds):
            if args.max_trajectories and i >= args.max_trajectories:
                break
            positions = features[0].to(device)
            particle_type = features[1].to(device)
            if has_material:
                material_property = features[2].to(device)
                n_particles = int(features[3])
            else:
                material_property = None
                n_particles = int(features[2])

            nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH

            mse_per_step, edge_counts = adaptive_rollout_mse(
                simulator, positions, particle_type, material_property,
                n_particles, nsteps, device,
                args.sigma_percentile, args.radius_factor,
            )

            mse_all_trajectories.append(mse_per_step)
            edge_counts_all.append(edge_counts)

            for s in EVAL_STEPS:
                idx = s - 1
                if idx < len(mse_per_step):
                    mse_at_steps[s].append(float(mse_per_step[idx]))

            n_eval += 1
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1} trajectories...")

    max_steps = max(len(m) for m in mse_all_trajectories)
    mse_stacked = np.full((n_eval, max_steps), np.nan)
    for i, m in enumerate(mse_all_trajectories):
        mse_stacked[i, :len(m)] = m

    steps = np.arange(1, max_steps + 1, dtype=np.int32)
    mse_mean = np.nanmean(mse_stacked, axis=0)
    mse_std = np.nanstd(mse_stacked, axis=0)
    n_at_step = np.sum(~np.isnan(mse_stacked), axis=0)

    # Edge count stats across all trajectories and steps
    edge_counts_stacked = np.full((n_eval, max_steps), np.nan)
    for i, ec in enumerate(edge_counts_all):
        edge_counts_stacked[i, :len(ec)] = ec
    mean_edges_per_step = float(np.nanmean(edge_counts_stacked))
    mean_edges_by_step = np.nanmean(edge_counts_stacked, axis=0)

    np.savez(
        output_stem + '.npz',
        steps=steps,
        mse_mean=mse_mean,
        mse_std=mse_std,
        mse_per_trajectory=mse_stacked,
        n_trajectories=n_eval,
        n_at_step=n_at_step,
        sigma_percentile=args.sigma_percentile,
        radius_factor=args.radius_factor,
        mean_edges_per_step=mean_edges_per_step,
        mean_edges_by_step=mean_edges_by_step,
        edge_counts_per_trajectory=edge_counts_stacked,
    )
    summary = {
        'data_path': data_path,
        'model_file': model_file,
        'sigma_percentile': args.sigma_percentile,
        'radius_factor': args.radius_factor,
        'n_trajectories': n_eval,
        'max_steps': int(max_steps),
        'mean_edges_per_step': mean_edges_per_step,
        'mse_at_steps': {s: float(np.mean(mse_at_steps[s])) if mse_at_steps[s] else None
                         for s in EVAL_STEPS},
    }
    with open(output_stem + '.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-step MSE to {output_stem}.npz and {output_stem}.json")
    print("\n" + "=" * 65)
    print("Adaptive Rollout MSE")
    print(f"  sigma_percentile={args.sigma_percentile}  radius_factor={args.radius_factor}")
    print(f"  Mean edges/step: {mean_edges_per_step:.1f}")
    print("=" * 65)
    print(f"{'Step':>6}  {'MSE (mean)':>14}  {'MSE (std)':>12}")
    print("-" * 65)
    for s in EVAL_STEPS:
        vals = mse_at_steps[s]
        if vals:
            print(f"{s:>6}  {np.mean(vals):>14.6e}  {np.std(vals):>12.6e}")
        else:
            print(f"{s:>6}  (no data)")
    print("=" * 65)


if __name__ == '__main__':
    main()
