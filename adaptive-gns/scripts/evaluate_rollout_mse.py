#!/usr/bin/env python3
"""
Evaluate rollout MSE at every step; save for graphing.
Week 1, Step 2: Record fixed-radius baseline numbers.
"""
import os
import sys
import argparse
import json

import numpy as np
import torch

# Add parent for gns imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns import learned_simulator, data_loader, reading_utils

INPUT_SEQUENCE_LENGTH = 6
KINEMATIC_PARTICLE_ID = 3
EVAL_STEPS = [1, 10, 50, 200]  # Summary steps printed to console


def rollout_with_step_mse(simulator, positions, particle_type, material_property,
                          n_particles_per_example, nsteps, device):
    """Rollout and return per-step MSE and per-step edge counts.

    Edge counts are captured from the radius graph the simulator actually
    consumes at every step (so the number is the true cost of inference, and
    is comparable to `evaluate_adaptive_rollout.py`'s `mean_edges_per_step`).
    """
    initial_positions = positions[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth = positions[:, INPUT_SEQUENCE_LENGTH:]
    current = initial_positions
    predictions = []
    edge_counts = []

    kinematic_mask = (particle_type == KINEMATIC_PARTICLE_ID).bool()
    non_kinematic = ~kinematic_mask

    for step in range(nsteps):
        if material_property is not None:
            node_features, edge_index, edge_features = simulator._encoder_preprocessor(
                current,
                nparticles_per_example=[n_particles_per_example],
                particle_types=particle_type,
                material_property=material_property,
            )
        else:
            node_features, edge_index, edge_features = simulator._encoder_preprocessor(
                current,
                nparticles_per_example=[n_particles_per_example],
                particle_types=particle_type,
            )
        edge_counts.append(int(edge_index.shape[1]))

        # Adaptive-gns simulator's _encode_process_decode returns (acc, variance).
        pred_acc, _ = simulator._encode_process_decode(node_features, edge_index, edge_features)
        next_pos = simulator._decoder_postprocessor(pred_acc, current)

        gt_step = ground_truth[:, step]
        next_pos = torch.where(
            kinematic_mask[:, None].expand(-1, next_pos.shape[-1]),
            gt_step, next_pos,
        )
        predictions.append(next_pos)
        current = torch.cat([current[:, 1:], next_pos[:, None, :]], dim=1)

    predictions = torch.stack(predictions)
    gt = ground_truth.permute(1, 0, 2)

    se = (predictions - gt) ** 2
    se_masked = se * non_kinematic[None, :, None].float()
    n_non_kinematic = non_kinematic.sum().item()
    mse_per_step = se_masked.sum(dim=(1, 2)) / (n_non_kinematic * predictions.shape[-1])

    return mse_per_step.cpu().numpy(), np.array(edge_counts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Dataset dir (e.g. ../Sand/dataset/)')
    parser.add_argument('--model_path', required=True, help='Model dir')
    parser.add_argument('--model_file', default='latest', help='Model checkpoint (e.g. model-100000.pt)')
    parser.add_argument('--max_trajectories', type=int, default=None, help='Limit eval to N trajectories')
    parser.add_argument('--connectivity_radius', type=float, default=None,
                        help='Override connectivity radius (default: use metadata value)')
    parser.add_argument('--output', '-o', default=None,
                        help='Save per-step MSE to .npz and .json (default: <model_path>/rollout_mse_<model_name>.npz)')
    args = parser.parse_args()

    data_path = args.data_path.rstrip('/') + '/'
    model_path = args.model_path.rstrip('/') + '/'

    # Resolve latest
    if args.model_file == 'latest':
        import glob
        candidates = glob.glob(os.path.join(model_path, 'model-*.pt'))
        if not candidates:
            raise FileNotFoundError(f"No model checkpoints in {model_path}")
        model_file = max(candidates, key=lambda p: int(p.split('-')[-1].split('.')[0]))
        model_file = os.path.basename(model_file)
    else:
        model_file = args.model_file

    output_path = args.output
    if output_path is None:
        model_basename = model_file.replace('.pt', '')
        if args.connectivity_radius is not None:
            output_path = os.path.join(model_path, f'rollout_mse_{model_basename}_r{args.connectivity_radius}')
        else:
            output_path = os.path.join(model_path, f'rollout_mse_{model_basename}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = reading_utils.read_metadata(data_path, 'rollout')
    if args.connectivity_radius is not None:
        print(f"Overriding connectivity radius: {metadata['default_connectivity_radius']} → {args.connectivity_radius}")
        metadata['default_connectivity_radius'] = args.connectivity_radius
    # Use train.py's _get_simulator logic for consistency
    from gns.train import _get_simulator
    simulator = _get_simulator(metadata, acc_noise_std=0.0, vel_noise_std=0.0, device=device)
    simulator.load(os.path.join(model_path, model_file))
    simulator.to(device)
    simulator.eval()

    ds = data_loader.get_data_loader_by_trajectories(path=data_path + 'test.npz')
    has_material = len(ds.dataset._data[0]) == 3

    mse_at_steps = {s: [] for s in EVAL_STEPS}
    mse_all_trajectories = []  # list of (nsteps,) arrays
    edge_counts_all = []        # list of (nsteps,) arrays

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

            mse_per_step, edge_counts = rollout_with_step_mse(
                simulator, positions, particle_type, material_property,
                torch.tensor([n_particles], dtype=torch.int32).to(device),
                nsteps, device,
            )

            mse_all_trajectories.append(mse_per_step)
            edge_counts_all.append(edge_counts)

            for s in EVAL_STEPS:
                step_idx = s - 1
                if step_idx < len(mse_per_step):
                    mse_at_steps[s].append(float(mse_per_step[step_idx]))

            n_eval += 1
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1} trajectories...")

    # Aggregate per-step MSE across trajectories (handle variable length)
    max_steps = max(len(m) for m in mse_all_trajectories)
    mse_stacked = np.full((n_eval, max_steps), np.nan)
    for i, m in enumerate(mse_all_trajectories):
        mse_stacked[i, :len(m)] = m

    edge_counts_stacked = np.full((n_eval, max_steps), np.nan)
    for i, ec in enumerate(edge_counts_all):
        edge_counts_stacked[i, :len(ec)] = ec
    mean_edges_per_step = float(np.nanmean(edge_counts_stacked))
    mean_edges_by_step = np.nanmean(edge_counts_stacked, axis=0)

    steps = np.arange(1, max_steps + 1, dtype=np.int32)
    mse_mean = np.nanmean(mse_stacked, axis=0)
    mse_std = np.nanstd(mse_stacked, axis=0)
    n_at_step = np.sum(~np.isnan(mse_stacked), axis=0)

    base = output_path
    for ext in ('.npz', '.json'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    np.savez(
        base + '.npz',
        steps=steps,
        mse_mean=mse_mean,
        mse_std=mse_std,
        mse_per_trajectory=mse_stacked,
        n_trajectories=n_eval,
        n_at_step=n_at_step,
        mean_edges_per_step=mean_edges_per_step,
        mean_edges_by_step=mean_edges_by_step,
        edge_counts_per_trajectory=edge_counts_stacked,
    )
    summary = {
        'data_path': data_path,
        'model_file': model_file,
        'n_trajectories': n_eval,
        'max_steps': int(max_steps),
        'mean_edges_per_step': mean_edges_per_step,
        'mse_at_steps': {s: float(np.mean(mse_at_steps[s])) if mse_at_steps[s] else None
                         for s in EVAL_STEPS},
    }
    with open(base + '.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved per-step MSE to {base}.npz and {base}.json")

    print("\n" + "=" * 60)
    print("Rollout MSE at specified steps (fixed-radius baseline)")
    print("=" * 60)
    print(f"Dataset: {data_path}")
    print(f"Model: {model_file}")
    print(f"Trajectories: {n_eval}")
    print(f"Mean edges/step (rollout-averaged): {mean_edges_per_step:.1f}")
    print("-" * 60)
    print(f"{'Step':>6}  {'MSE (mean)':>14}  {'MSE (std)':>12}")
    print("-" * 60)
    for s in EVAL_STEPS:
        vals = mse_at_steps[s]
        if vals:
            mean_mse = np.mean(vals)
            std_mse = np.std(vals)
            print(f"{s:>6}  {mean_mse:>14.6e}  {std_mse:>12.6e}")
        else:
            print(f"{s:>6}  (no data)")
    print("=" * 60)


if __name__ == '__main__':
    main()
