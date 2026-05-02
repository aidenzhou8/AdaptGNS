#!/usr/bin/env python3
"""
Convert DeepMind GNS TFRecord dataset files to the .npz format expected by gns-main.

The original DeepMind GNS datasets (WaterDrop, etc.) are stored as TFRecords of
tf.train.SequenceExamples with:
  context_features:  particle_type  (int64 list)
  sequence_features: position       (one float32 byte-string per timestep)

This script reads train/valid/test TFRecords and writes the equivalent .npz files
that gns/data_loader.py can consume directly.

Requirements:
  pip install tensorflow-cpu  (only needed for this script)

Usage:
  python convert_tfrecord_to_npz.py \
    --input_dir  /path/to/WaterDrop/dataset \
    --output_dir /path/to/WaterDrop/dataset \
    --splits train valid test \
    --ndim 2
"""
import argparse
import os
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir',  required=True,
                   help='Directory containing *.tfrecord files and metadata.json')
    p.add_argument('--output_dir', default=None,
                   help='Where to write .npz files (default: same as input_dir)')
    p.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'])
    p.add_argument('--ndim', type=int, default=2, help='Spatial dimensions (default: 2)')
    return p.parse_args()


def load_split(tfrecord_path, ndim):
    """Read all trajectories from a TFRecord file.

    Returns a list of (positions, particle_types) tuples where:
      positions:      np.ndarray float32  (sequence_length, n_particles, ndim)
      particle_types: np.ndarray int64    (n_particles,)
    """
    import tensorflow as tf

    context_desc = {
        'particle_type': tf.io.VarLenFeature(tf.string),
    }
    seq_desc = {
        'position': tf.io.VarLenFeature(tf.string),
    }

    trajectories = []
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for raw in dataset:
        ctx, seq = tf.io.parse_single_sequence_example(
            raw,
            context_features=context_desc,
            sequence_features=seq_desc,
        )

        # particle_type: serialized int64 array -> (n_particles,)
        pt_bytes = tf.sparse.to_dense(ctx['particle_type'], default_value=b'')[0]
        particle_type = np.frombuffer(pt_bytes.numpy(), dtype=np.int64)

        # position: 2D dense tensor (nsteps, 1) where each cell is a
        # serialized float32 array of shape (n_particles * ndim,)
        pos_sparse = tf.sparse.to_dense(seq['position'], default_value=b'')
        nsteps = pos_sparse.shape[0]
        pos_list = []
        for i in range(nsteps):
            arr = np.frombuffer(pos_sparse[i][0].numpy(), dtype=np.float32)
            pos_list.append(arr.reshape(-1, ndim))          # (n_particles, ndim)
        positions = np.stack(pos_list, axis=0)              # (nsteps, n_particles, ndim)

        trajectories.append((positions, particle_type))

    return trajectories


def save_npz(trajectories, output_path):
    """Save list of (positions, particle_type) as a single .npz file."""
    data = {}
    for i, (pos, pt) in enumerate(trajectories):
        data[str(i)] = (pos, pt)
    np.savez_compressed(output_path, **{str(i): v for i, v in enumerate(data.values())})
    print(f"  Saved {len(trajectories)} trajectories → {output_path}")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    for split in args.splits:
        tfrecord = os.path.join(args.input_dir, f'{split}.tfrecord')
        if not os.path.exists(tfrecord):
            print(f"  [skip] {tfrecord} not found")
            continue
        print(f"Reading {tfrecord} ...")
        trajectories = load_split(tfrecord, args.ndim)
        print(f"  {len(trajectories)} trajectories loaded")
        out = os.path.join(output_dir, f'{split}.npz')
        # gns data_loader expects list of (positions, particle_type) tuples
        np.savez_compressed(
            out,
            **{str(i): np.array([pos, pt], dtype=object)
               for i, (pos, pt) in enumerate(trajectories)}
        )
        print(f"  Saved → {out}")

    print("Done. Copy metadata.json to the output_dir if it isn't already there.")


if __name__ == '__main__':
    main()
