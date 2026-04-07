"""
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from collections import defaultdict
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


def build_h5_paths(base_path: Path, replica: int, n_steps: int) -> list[Path]:
    """Return expected HDF5 segment paths for one replica."""
    return [
        base_path
        / f"rest2_{replica:04d}_{segment:04d}"
        / f"rest2_{replica:04d}_{segment:04d}"
        / "replica_00.h5"
        for segment in range(n_steps)
    ]


def load_replica_trajectory(
    base_path: Path,
    replica: int,
    n_steps: int,
    top_file: Path,
    stride: int = 1,
):
    """
    Load multiple HDF5 trajectory segments, skip unreadable files, and join them.
    """
    paths = build_h5_paths(base_path, replica, n_steps)
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
        print("Skipped unreadable segments:")
        for path, message in bad:
            print(f" - {path}: {message}")

    print(f"Total loaded frames: {joined.n_frames}")
    return joined


def load_noe_csv(csv_path: Path) -> pd.DataFrame:
    """Load NOE definitions from CSV."""
    required_columns = [
        "selection_1",
        "selection_2",
        "target",
        "lower_tol",
        "upper_tol",
        "group",
    ]

    if not csv_path.exists():
        raise FileNotFoundError(f"NOE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"NOE CSV {csv_path} is missing required columns: {missing_columns}"
        )

    return df[required_columns].copy()


def compute_noe_averages(traj, noe_table: pd.DataFrame, output_file: Path) -> None:
    """
    Compute NOE distances, average them by group using r^-6 averaging,
    and save the grouped results to CSV.
    """
    atom_pairs = []
    group_to_indices = defaultdict(list)

    for i, (_, row) in enumerate(noe_table.iterrows()):
        selection_1 = row["selection_1"]
        selection_2 = row["selection_2"]

        atom_1 = traj.topology.select(selection_1)
        atom_2 = traj.topology.select(selection_2)

        if len(atom_1) != 1 or len(atom_2) != 1:
            raise ValueError(
                f"Selections must return exactly one atom each, got "
                f"{len(atom_1)} and {len(atom_2)} for "
                f"{selection_1!r} and {selection_2!r}"
            )

        atom_pairs.append([atom_1[0], atom_2[0]])
        group_to_indices[int(row["group"])].append(i)

    atom_pairs = np.array(atom_pairs, dtype=int)
    distances_nm = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
    distances_A = distances_nm * 10.0

    results = []
    for group in sorted(group_to_indices):
        idx = group_to_indices[group]

        group_rows = noe_table[noe_table["group"] == group]
        first_row = group_rows.iloc[0]

        flat_distances = distances_A[:, idx].reshape(-1)
        r6_avg = np.mean(flat_distances**-6) ** (-1.0 / 6.0)

        target = float(first_row["target"])
        lower_tol = float(first_row["lower_tol"])
        upper_tol = float(first_row["upper_tol"])
        lower_limit = target - lower_tol
        upper_limit = target + upper_tol
        violation = r6_avg > upper_limit

        results.append(
            {
                "group": group,
                "group_id_1based": group + 1,
                "target_A": target,
                "lower_bound_A": lower_tol,
                "upper_bound_A": upper_tol,
                "lower_limit_A": lower_limit,
                "upper_limit_A": upper_limit,
                "r6_avg_A": float(r6_avg),
                "violation": bool(violation),
                "n_pairs": len(idx),
            }
        )

    df = pd.DataFrame(results)

    violation_pct = 100.0 * df["violation"].mean()
    print(f"Violation percentage: {violation_pct:.1f}%")
    print(f"Saving averaged distances to {output_file}...")
    df.to_csv(output_file, index=False)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute grouped TRP NOE r^-6 averages from REST2 HDF5 segments."
    )
    parser.add_argument(
        "--molecule",
        required=True,
        choices=["TRP", "TRPu"],
        help="Molecule to analyze.",
    )
    parser.add_argument(
        "--topology", required=True, type=Path, help="Topology PDB file."
    )
    parser.add_argument(
        "--base-path", required=True, type=Path, help="Base trajectory directory."
    )
    parser.add_argument("--replica", type=int, default=0, help="Replica index.")
    parser.add_argument(
        "--n-steps", required=True, type=int, help="Number of HDF5 segments."
    )
    parser.add_argument(
        "--stride", type=int, default=10, help="Stride used when loading trajectories."
    )
    parser.add_argument(
        "--noe-csv",
        required=True,
        type=Path,
        help="CSV file containing NOE definitions.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output CSV file.")
    return parser


def main() -> int:
    """Run the CLI."""
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing molecule: {args.molecule}")

    traj = load_replica_trajectory(
        base_path=args.base_path,
        replica=args.replica,
        n_steps=args.n_steps,
        top_file=args.topology,
        stride=args.stride,
    )

    noe_table = load_noe_csv(args.noe_csv)
    compute_noe_averages(traj, noe_table, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
