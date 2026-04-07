""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


def all_unique_pairs(n_atoms: int) -> np.ndarray:
    """Return all unique atom pairs i < j for a given number of atoms."""
    upper = np.triu_indices(n_atoms, 1)
    return np.stack(upper, axis=1)


def compute_rmsd_dist(traj, ref, selection: str = "not element H") -> np.ndarray:
    """
    Compute RMSDdist as the root-mean-squared difference of interatomic distances.

    This metric is alignment-free and is computed on the selected atoms.
    """
    selected = traj.topology.select(selection)
    traj_selected = traj.atom_slice(selected)
    ref_selected = ref.atom_slice(selected)

    pairs = all_unique_pairs(len(selected))
    ref_distances = md.compute_distances(ref_selected, pairs)[0]
    traj_distances = md.compute_distances(traj_selected, pairs)

    squared_diff = (traj_distances - ref_distances) ** 2
    return np.sqrt(np.mean(squared_diff, axis=1))


def build_h5_paths(
    base_path: Path, replica: int, n_steps: int, campaign: int
) -> list[Path]:
    """Return expected HDF5 paths for one replica and campaign."""
    return [
        base_path
        / f"rest2_{campaign:04d}_{segment:04d}"
        / f"rest2_{campaign:04d}_{segment:04d}"
        / f"replica_{replica:02d}.h5"
        for segment in range(n_steps)
    ]


def load_replica_trajectory(
    pdb_top: Path,
    base_path: Path,
    replica: int,
    n_steps: int,
    stride: int = 1,
    campaign: int = 0,
):
    """Load and join all HDF5 trajectory segments for one replica."""
    paths = build_h5_paths(
        base_path=base_path, replica=replica, n_steps=n_steps, campaign=campaign
    )
    trajectories = [
        md.load(str(path), top=str(pdb_top), stride=stride) for path in paths
    ]

    joined = trajectories[0]
    for traj in trajectories[1:]:
        joined = joined.join(traj, check_topology=False)

    return joined


def compute_rmsd_ca(traj, ref, selection: str = "name CA") -> np.ndarray:
    """Compute CA RMSD against the reference structure."""
    indices = traj.topology.select(selection)
    return md.rmsd(
        traj, ref, atom_indices=indices, ref_atom_indices=indices, parallel=True
    )


def run_replica_analysis(
    molecule: str,
    pdb_top: Path,
    pdb_ref: Path,
    base_path: Path,
    replica: int,
    n_steps: int,
    stride: int,
    selection: str,
    campaign: int,
    outdir: Path,
) -> Path:
    """Run the full RMSD/Rg analysis for one replica and write a CSV file."""
    outdir.mkdir(parents=True, exist_ok=True)

    ref = md.load(str(pdb_ref))
    traj = load_replica_trajectory(
        pdb_top=pdb_top,
        base_path=base_path,
        replica=replica,
        n_steps=n_steps,
        stride=stride,
        campaign=campaign,
    )

    rmsd_dist_nm = compute_rmsd_dist(traj, ref, selection=selection)
    rmsd_ca = compute_rmsd_ca(traj, ref)

    protein_indices = traj.topology.select("protein")
    rg_nm = md.compute_rg(traj.atom_slice(protein_indices))

    dataframe = pd.DataFrame(
        {
            "replica": replica,
            "frame": np.arange(traj.n_frames, dtype=int),
            "rmsd_ca": rmsd_ca,
            "rmsd_dist_nm": rmsd_dist_nm,
            "rg_nm": rg_nm,
        }
    )

    output_file = outdir / f"{molecule}_replica_{replica:02d}.csv"
    dataframe.to_csv(output_file, index=False)

    print(
        f"[{molecule} | replica {replica}] wrote {len(dataframe)} frames to {output_file}"
    )
    return output_file


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute RMSDdist, CA RMSD, and radius of gyration for a single replica."
    )
    parser.add_argument(
        "--molecule", required=True, help="Name of the analyzed molecule."
    )
    parser.add_argument(
        "--pdb-top", required=True, type=Path, help="Topology PDB file."
    )
    parser.add_argument(
        "--pdb-ref", required=True, type=Path, help="Reference PDB file."
    )
    parser.add_argument(
        "--base-path",
        required=True,
        type=Path,
        help="Base directory containing trajectory segments.",
    )
    parser.add_argument("--replica", required=True, type=int, help="Replica index.")
    parser.add_argument(
        "--n-steps", required=True, type=int, help="Number of trajectory segments."
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Stride used when loading trajectories."
    )
    parser.add_argument(
        "--selection", default="not element H", help="Atom selection used for RMSDdist."
    )
    parser.add_argument("--campaign", type=int, default=0, help="Campaign index.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("csv"), help="Output directory."
    )
    return parser


def main() -> int:
    """Run the CLI."""
    args = build_parser().parse_args()

    run_replica_analysis(
        molecule=args.molecule,
        pdb_top=args.pdb_top,
        pdb_ref=args.pdb_ref,
        base_path=args.base_path,
        replica=args.replica,
        n_steps=args.n_steps,
        stride=args.stride,
        selection=args.selection,
        campaign=args.campaign,
        outdir=args.outdir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
