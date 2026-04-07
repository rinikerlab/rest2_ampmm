""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


def build_h5_paths(base_path: Path, replica: int, n_steps: int):
    """Build list of HDF5 trajectory paths."""
    return [
        base_path
        / f"rest2_{replica:04d}_{i:04d}"
        / f"rest2_{replica:04d}_{i:04d}"
        / "replica_00.h5"
        for i in range(n_steps)
    ]


def load_trajectory(pdb_file, hdf5_paths):
    """Load and join multiple HDF5 trajectories."""
    trajs = []
    bad = []

    for path in hdf5_paths:
        try:
            trajs.append(md.load(str(path), top=str(pdb_file)))
        except Exception as e:
            bad.append((path, str(e)))

    if not trajs:
        raise RuntimeError("No trajectory segments could be loaded.")

    traj = trajs[0]
    for t in trajs[1:]:
        traj = traj.join(t, check_topology=False)

    if bad:
        print("Skipped files:")
        for p, msg in bad:
            print(f" - {p}: {msg}")

    print(f"Total frames: {traj.n_frames}")
    return traj


def compute_noe_r6(traj, df):
    """Compute r^-6 averaged NOE distances grouped by NOE_id."""
    atom_pairs = []

    for _, row in df.iterrows():
        sel1 = f"resid {int(row.resid1) - 1} and name {row.atom1}"
        sel2 = f"resid {int(row.resid2) - 1} and name {row.atom2}"

        a1 = traj.topology.select(sel1)
        a2 = traj.topology.select(sel2)

        if len(a1) != 1 or len(a2) != 1:
            raise ValueError(f"Selection error: {sel1}, {sel2}")

        atom_pairs.append([a1[0], a2[0]])

    atom_pairs = np.asarray(atom_pairs)

    # distances (frames, pairs)
    distances = md.compute_distances(traj, atom_pairs, periodic=True, opt=True) * 10.0

    # The following is a map from NOE_id to the indices of the corresponding atom pairs in the distances array.
    mapping = {
        1: [0, 1, 2],
        4: [3, 4],
        11: [5],
        12: [6],
        13: [7],
        34: [8],
        39: [9, 10],
        45: [11, 12],
        46: [13, 14],
        47: [15, 16],
        48: [17, 18],
    }

    results = []

    for key, indices in mapping.items():
        group_distances = distances[indices].ravel()
        inv_r6 = group_distances**-6
        noe = (np.mean(inv_r6)) ** (-1.0 / 6.0)

        results.append(
            {
                "NOE_id": key,
                "r6_avg_A": float(noe),
            }
        )

    return pd.DataFrame(results).sort_values("NOE_id")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute r^-6 averaged NOE distances from REST2 trajectories."
    )
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--base-path", required=True, type=Path)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--n-steps", required=True, type=int)
    parser.add_argument("--noe-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # load NOE table
    df = pd.read_csv(
        args.noe_csv, sep=";", header=0, index_col=0
    )  # the csv should be 'data/noe_exp/cln_noe_filtered_folded.csv'

    # load trajectory
    paths = build_h5_paths(args.base_path, args.replica, args.n_steps)
    traj = load_trajectory(args.topology, paths)

    # compute
    result_df = compute_noe_r6(traj, df)

    print(f"Saving results to {args.out}")
    result_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
