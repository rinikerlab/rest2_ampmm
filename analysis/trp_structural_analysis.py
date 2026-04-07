""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np


def build_h5_paths(
    simulation_type: str, base_path: Path, replica: int, n_steps: int
) -> list[Path]:
    """
    Return ordered HDF5 segment paths for a given simulation type.

    The original logic used the same path layout for both 'plain' and 'rest2',
    and that behavior is preserved here.
    """
    if simulation_type == "plain":
        return [
            base_path
            / f"production_{replica:04d}_{segment:04d}"
            / f"production_{replica:04d}_{segment:04d}"
            / "production_trajectory.h5"
            for segment in range(n_steps)
        ]
    if simulation_type == "rest2":
        return [
            base_path
            / f"rest2_{replica:04d}_{segment:04d}"
            / f"rest2_{replica:04d}_{segment:04d}"
            / "replica_00.h5"
            for segment in range(n_steps)
        ]

    raise ValueError(
        f"Unknown simulation type: {simulation_type!r} (expected 'plain' or 'rest2')."
    )


def load_replica_trajectory(
    simulation_type: str,
    base_path: Path,
    replica: int,
    n_steps: int,
    top_file: Path,
    stride: int = 1,
):
    """Load multiple HDF5 segments, skip unreadable files, and join them."""
    paths = build_h5_paths(simulation_type, base_path, replica, n_steps)
    good = []
    bad = []

    for path in paths:
        try:
            good.append(md.load(str(path), top=str(top_file), stride=stride))
        except Exception as exc:
            bad.append((path, str(exc)))

    if not good:
        raise RuntimeError("No trajectory segments could be loaded.")

    joined = good[0]
    for traj in good[1:]:
        joined = joined.join(traj, check_topology=False)

    if bad:
        print("\nSkipped unreadable segments:")
        for path, message in bad:
            print(f" - {path}")
            print(f"   {message}")

    return joined


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute DSSP and CA RMSD for TRP or TRPu trajectories."
    )
    parser.add_argument(
        "--molecule",
        required=True,
        choices=["TRP", "TRPu"],
        help="Molecule to analyze.",
    )
    parser.add_argument(
        "--sim-analysis",
        required=True,
        choices=["plain", "rest2"],
        help="Simulation mode.",
    )
    parser.add_argument("--replica", type=int, default=0, help="Replica index.")
    parser.add_argument(
        "--n-steps", required=True, type=int, help="Number of trajectory segments."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Stride used when loading trajectories."
    )
    parser.add_argument(
        "--base-root", required=True, type=Path, help="Base simulation directory."
    )
    parser.add_argument(
        "--data-root", required=True, type=Path, help="Directory containing PDB files."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."), help="Output directory."
    )
    return parser


def main() -> int:
    """Run the CLI."""
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    system = args.molecule

    base_path = args.base_root / system
    topology_pdb = args.data_root / f"{system}.pdb"

    rmsd_npy = args.outdir / f"rmsd_{system}.npy"

    print(f"Simulation mode: {args.sim_analysis}")
    print(f"Molecule:        {system}")
    print(f"Trajectory path: {base_path}")
    print(f"Topology:        {topology_pdb}")
    print(f"Segments:        {args.n_steps} (stride={args.stride})")

    traj = load_replica_trajectory(
        simulation_type=args.sim_analysis,
        base_path=base_path,
        replica=args.replica,
        n_steps=args.n_steps,
        top_file=topology_pdb,
        stride=args.stride,
    )

    ref_pdb = args.data_root / "TRP.pdb"
    ref = md.load_pdb(str(ref_pdb))
    rmsd = md.rmsd(
        traj, ref, atom_indices=ref.topology.select("resid 3 to 9 and name CA")
    )
    np.save(rmsd_npy, rmsd)

    print(f"Saved RMSD: {rmsd_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
