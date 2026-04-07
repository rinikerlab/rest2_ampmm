""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mdtraj as md
except ImportError:
    md = None


def require_mdtraj() -> None:
    """Ensure MDTraj is installed before running the analysis."""
    if md is None:
        raise ImportError("mdtraj is required. Install it and run again.")


def build_rest2_h5_paths(base_path: Path, replica: int, n_steps: int) -> list[Path]:
    """Return expected REST2 HDF5 segment paths for one replica."""
    return [
        base_path
        / f"rest2_0000_{segment:04d}"
        / f"rest2_0000_{segment:04d}"
        / f"replica_{replica:02d}.h5"
        for segment in range(n_steps)
    ]


def load_rest2_replica_trajectory(
    pdb_file: Path,
    base_path: Path,
    replica: int,
    n_steps: int,
    stride: int = 1,
):
    """Load and join REST2 HDF5 segments for one replica."""
    require_mdtraj()

    paths = [
        path
        for path in build_rest2_h5_paths(base_path, replica, n_steps)
        if path.exists()
    ]
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files found for replica {replica} in {base_path}"
        )

    trajectories = []
    for path in paths:
        try:
            trajectories.append(md.load(str(path), top=str(pdb_file), stride=stride))
        except TypeError:
            trajectories.append(md.load(str(path), stride=stride))

    joined = trajectories[0]
    for traj in trajectories[1:]:
        joined = joined.join(traj, check_topology=False)

    return joined


def compute_phi_psi(
    traj, resid_start: int, resid_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute phi/psi angles in degrees for residues in the given MDTraj resid range.
    """
    require_mdtraj()

    selected_atoms = traj.topology.select(
        f"resid {resid_start} to {resid_end} and backbone"
    )
    if selected_atoms.size == 0:
        raise ValueError(
            f"Empty backbone selection for resid range {resid_start}..{resid_end}. "
            "Remember: MDTraj resid is 0-based."
        )

    sliced = traj.atom_slice(selected_atoms)
    _, phi = md.compute_phi(sliced)
    _, psi = md.compute_psi(sliced)

    return np.degrees(phi), np.degrees(psi)


def save_rest2_ramachandran_csv(
    base_path: Path,
    pdb_file: Path,
    n_replicas: int,
    n_steps: int,
    stride: int,
    output_csv: Path,
    resid_start: int,
    resid_end: int,
) -> None:
    """Compute and save phi/psi values for all REST2 replicas."""
    require_mdtraj()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for replica in range(n_replicas):
        traj = load_rest2_replica_trajectory(
            pdb_file=pdb_file,
            base_path=base_path,
            replica=replica,
            n_steps=n_steps,
            stride=stride,
        )
        phi, psi = compute_phi_psi(traj, resid_start=resid_start, resid_end=resid_end)
        rows.append(
            np.column_stack(
                [np.full(phi.size, replica, dtype=int), phi.ravel(), psi.ravel()]
            )
        )

    data = np.vstack(rows)
    np.savetxt(
        output_csv,
        data,
        fmt=["%d", "%.6f", "%.6f"],
        delimiter=",",
        header="replica,phi_deg,psi_deg",
        comments="",
    )
    print(f"Saved REST2 phi/psi for {n_replicas} replicas to {output_csv}")


def save_plain_ramachandran_csv(
    base_path: Path,
    pdb_file: Path,
    output_csv: Path,
    resid_start: int,
    resid_end: int,
    stride: int = 10,
) -> None:
    """Compute and save phi/psi values for plain MD DCD trajectories."""
    require_mdtraj()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dcd_files = sorted(base_path.glob("production_00*_00*/production_00*_00*/traj.dcd"))
    if not dcd_files:
        raise FileNotFoundError(
            f"No trajectories found under pattern: {base_path / 'production_00*_00*/production_00*_00*/traj.dcd'}"
        )

    traj = md.join(
        [md.load(str(file), top=str(pdb_file), stride=stride) for file in dcd_files]
    )
    phi, psi = compute_phi_psi(traj, resid_start=resid_start, resid_end=resid_end)

    np.savetxt(
        output_csv,
        np.column_stack([phi.ravel(), psi.ravel()]),
        fmt="%.6f",
        delimiter=",",
        header="phi_deg,psi_deg",
        comments="",
    )
    print(f"Saved {traj.n_frames} phi/psi points to {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute Ramachandran phi/psi angles and save them to CSV."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    rest2 = subparsers.add_parser("rest2", help="Analyze REST2 HDF5 input.")
    rest2.add_argument(
        "--molecule", required=True, help="Name of the analyzed molecule."
    )
    rest2.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Base directory containing REST2 segments.",
    )
    rest2.add_argument("--pdb", required=True, type=Path, help="Topology PDB file.")
    rest2.add_argument("--n-rep", required=True, type=int, help="Number of replicas.")
    rest2.add_argument(
        "--n-steps", required=True, type=int, help="Number of HDF5 segments."
    )
    rest2.add_argument(
        "--stride", type=int, default=1, help="Stride used when loading segments."
    )
    rest2.add_argument("--out", required=True, type=Path, help="Output CSV file.")
    rest2.add_argument(
        "--resid-start", type=int, default=1, help="Start MDTraj resid (0-based)."
    )
    rest2.add_argument(
        "--resid-end", type=int, default=13, help="End MDTraj resid (0-based)."
    )

    plain = subparsers.add_parser("plain", help="Analyze plain MD DCD input.")
    plain.add_argument(
        "--molecule", required=True, help="Name of the analyzed molecule."
    )
    plain.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Base directory containing production segments.",
    )
    plain.add_argument("--pdb", required=True, type=Path, help="Topology PDB file.")
    plain.add_argument(
        "--stride", type=int, default=1, help="Stride used when loading trajectories."
    )
    plain.add_argument("--out", required=True, type=Path, help="Output CSV file.")
    plain.add_argument(
        "--resid-start", type=int, default=1, help="Start MDTraj resid (0-based)."
    )
    plain.add_argument(
        "--resid-end", type=int, default=13, help="End MDTraj resid (0-based)."
    )

    return parser


def main() -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if md is None:
        print("ERROR: mdtraj is not installed in this environment.", file=sys.stderr)
        return 2

    if args.mode == "rest2":
        save_rest2_ramachandran_csv(
            base_path=args.path,
            pdb_file=args.pdb,
            n_replicas=args.n_rep,
            n_steps=args.n_steps,
            stride=args.stride,
            output_csv=args.out,
            resid_start=args.resid_start,
            resid_end=args.resid_end,
        )
    elif args.mode == "plain":
        save_plain_ramachandran_csv(
            base_path=args.path,
            pdb_file=args.pdb,
            output_csv=args.out,
            resid_start=args.resid_start,
            resid_end=args.resid_end,
            stride=args.stride,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
