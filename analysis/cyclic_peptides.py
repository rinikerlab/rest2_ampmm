""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from collections import defaultdict
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

N_REST2_STEPS = 100
N_PROD_STEPS = 50


def build_rest2_paths(base_path: Path, replica: int, n_steps: int) -> list[Path]:
    """Return expected REST2 segment paths."""
    return [
        base_path
        / f"rest2_{replica:04d}_{segment:04d}"
        / f"rest2_{replica:04d}_{segment:04d}"
        / "replica_00.h5"
        for segment in range(n_steps)
    ]


def build_production_paths(base_path: Path, n_steps: int) -> list[Path]:
    """Return expected production segment paths."""
    return [
        base_path
        / f"production_0000_{segment:04d}"
        / f"production_0000_{segment:04d}"
        / "production_trajectory.h5"
        for segment in range(n_steps)
    ]


def load_segmented_trajectory(paths: list[Path], top_file: Path, stride: int = 1):
    """Load readable trajectory segments and join them."""
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

    return joined


def is_hydrogen(atom) -> bool:
    """Return True if the atom is a hydrogen."""
    return (
        atom.element is not None and atom.element.symbol == "H"
    ) or atom.name.upper().startswith("H")


def build_neighbors(topology) -> dict[int, set[int]]:
    """Build a bond-neighbor map for the topology."""
    neighbors = {atom.index: set() for atom in topology.atoms}
    for bond in topology.bonds:
        i = bond.atom1.index
        j = bond.atom2.index
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def residue_atom(residue, name: str):
    """Return the atom with the requested name from a residue."""
    for atom in residue.atoms:
        if atom.name == name:
            return atom
    raise KeyError(f"Atom {name} not found in residue {residue}")


def bonded_hydrogens(atom, topology, neighbors) -> list:
    """Return hydrogen atoms bonded to the given atom."""
    hydrogens = []
    for index in neighbors[atom.index]:
        neighbor = topology.atom(index)
        if is_hydrogen(neighbor):
            hydrogens.append(neighbor)
    return hydrogens


def pick_h_atoms(residue, code: str, topology, neighbors) -> list[int]:
    """
    Return hydrogen atom indices corresponding to a shorthand proton code.

    Supported codes:
    - HN: hydrogens attached to backbone N
    - HA: hydrogens attached to CA
    - HB: hydrogens attached to CB
    """
    code = code.strip().lower()

    if code == "hn":
        atom = residue_atom(residue, "N")
        return [h.index for h in bonded_hydrogens(atom, topology, neighbors)]

    if code == "ha":
        atom = residue_atom(residue, "CA")
        return [h.index for h in bonded_hydrogens(atom, topology, neighbors)]

    if code == "hb":
        atom = residue_atom(residue, "CB")
        return [h.index for h in bonded_hydrogens(atom, topology, neighbors)]

    raise ValueError(f"Unknown atom code: {code}")


def parse_label(label: str) -> tuple[str, str]:
    """Split labels such as 'Ser HN' into residue name and atom code."""
    residue_name, atom_code = label.strip().split()
    return residue_name.strip().capitalize()[:3], atom_code.strip()


def build_residue_map_lists(topology) -> dict[str, list]:
    """Map each residue short name to a list of matching residues."""
    residue_map = defaultdict(list)
    for residue in topology.residues:
        short_name = residue.name.strip().capitalize()[:3]
        residue_map[short_name].append(residue)
    return residue_map


def compute_noe_r6avg(
    traj, label1: str, label2: str, residue_map, neighbors, eps: float = 1e-12
):
    """
    Compute r^-6 averaged NOE distance between two labels.

    Labels are interpreted across all residues of the matching residue type.
    For each label, all hydrogens attached to the relevant heavy atom are selected.
    All pairwise distances between the two selected hydrogen sets are computed,
    concatenated across frames and equivalent pairs, and then used for statistics
    and r^-6 averaging.
    """
    topology = traj.topology
    residue1, code1 = parse_label(label1)
    residue2, code2 = parse_label(label2)

    if residue1 not in residue_map or residue2 not in residue_map:
        return None, {
            "label1": label1,
            "label2": label2,
            "reason": "residue name not in topology",
        }

    indices1 = []
    indices2 = []

    for residue in residue_map[residue1]:
        try:
            indices1.extend(pick_h_atoms(residue, code1, topology, neighbors))
        except Exception:
            pass

    for residue in residue_map[residue2]:
        try:
            indices2.extend(pick_h_atoms(residue, code2, topology, neighbors))
        except Exception:
            pass

    if len(indices1) == 0 or len(indices2) == 0:
        return None, {
            "label1": label1,
            "label2": label2,
            "idx1": indices1,
            "idx2": indices2,
        }

    pairs = np.array([[i, j] for i in indices1 for j in indices2], dtype=int)
    distances_nm = md.compute_distances(traj, pairs)
    distances_a = (distances_nm * 10.0).ravel()

    r6 = np.mean(1.0 / np.maximum(distances_a, eps) ** 6)
    r6_average = r6 ** (-1.0 / 6.0)

    stats = {
        "distance_A_r6avg": float(r6_average),
        "distance_A_mean": float(np.mean(distances_a)),
        "distance_A_median": float(np.median(distances_a)),
        "distance_A_min": float(np.min(distances_a)),
        "distance_A_max": float(np.max(distances_a)),
        "n_pairs_used": int(pairs.shape[0]),
        "n_distances_used": int(distances_a.size),
    }
    return stats, None


def compute_noe_table(traj, noe_exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute NOE distances for all rows in the experimental NOE table."""
    topology = traj.topology
    neighbors = build_neighbors(topology)
    residue_map = build_residue_map_lists(topology)

    output_rows = []
    missing_rows = []

    for _, row in noe_exp.iterrows():
        stats, missing = compute_noe_r6avg(
            traj,
            row["atom1"],
            row["atom2"],
            residue_map=residue_map,
            neighbors=neighbors,
        )

        if missing is not None:
            missing_rows.append({**row.to_dict(), **missing})
            output = {
                **row.to_dict(),
                "distance_A_r6avg": np.nan,
                "distance_A_mean": np.nan,
                "distance_A_median": np.nan,
                "distance_A_min": np.nan,
                "distance_A_max": np.nan,
                "n_pairs_used": 0,
                "n_distances_used": 0,
            }
        else:
            output = {**row.to_dict(), **stats}

        output_rows.append(output)

    return pd.DataFrame(output_rows), pd.DataFrame(missing_rows)


def atom_index(residue, name: str) -> int:
    """Return the index of a named atom within a residue."""
    return residue_atom(residue, name).index


def phi_psi_df_cyclic(traj, residue_indices: list[int] | None = None) -> pd.DataFrame:
    """Compute cyclic phi/psi values for all selected residues."""
    topology = traj.topology
    residues = list(topology.residues)

    if residue_indices is None:
        residue_indices = [residue.index for residue in residues]

    residue_indices = [index for index in residue_indices if 0 <= index < len(residues)]
    n_residues = len(residue_indices)
    if n_residues < 3:
        raise ValueError("At least 3 residues are required to define cyclic phi/psi.")

    phi_quads = []
    psi_quads = []
    kept_residues = []

    for k, residue_index in enumerate(residue_indices):
        prev_index = residue_indices[(k - 1) % n_residues]
        next_index = residue_indices[(k + 1) % n_residues]

        residue_prev = residues[prev_index]
        residue_curr = residues[residue_index]
        residue_next = residues[next_index]

        try:
            c_prev = atom_index(residue_prev, "C")
            n_curr = atom_index(residue_curr, "N")
            ca_curr = atom_index(residue_curr, "CA")
            c_curr = atom_index(residue_curr, "C")
            n_next = atom_index(residue_next, "N")
        except Exception:
            continue

        phi_quads.append([c_prev, n_curr, ca_curr, c_curr])
        psi_quads.append([n_curr, ca_curr, c_curr, n_next])
        kept_residues.append(residue_index)

    if not kept_residues:
        raise ValueError("No residues contained the required N/CA/C atoms.")

    phi = np.degrees(md.compute_dihedrals(traj, np.array(phi_quads, dtype=int)))
    psi = np.degrees(md.compute_dihedrals(traj, np.array(psi_quads, dtype=int)))

    frames = np.arange(traj.n_frames)
    frames_data = []
    for column, residue_index in enumerate(kept_residues):
        frames_data.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "res": residue_index,
                    "name": topology.residue(residue_index).name,
                    "phi": phi[:, column],
                    "psi": psi[:, column],
                }
            )
        )

    return pd.concat(frames_data, ignore_index=True)


def load_noe_csv(csv_path: Path) -> pd.DataFrame:
    """Load experimental NOE restraints from a CSV file."""
    required_columns = ["noe_id", "atom1", "atom2", "nmr", "lower_bound", "upper_bound"]

    if not csv_path.exists():
        raise FileNotFoundError(f"NOE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"NOE CSV {csv_path} is missing required columns: {missing_columns}"
        )

    return df[required_columns].copy()


def build_system_config(base_root: Path, noe_dir: Path) -> dict[str, dict]:
    """Return hard-coded configuration for CYC2 and CYC2t."""
    return {
        "CYC2": {
            "sim_path": base_root / "CYC2",
            "rest2_steps": N_REST2_STEPS,
            "prod_steps": N_PROD_STEPS,
            "noe_csv": noe_dir / "noe_CYC2.csv",
        },
        "CYC2t": {
            "sim_path": base_root / "CYC2t",
            "rest2_steps": N_REST2_STEPS,
            "prod_steps": N_PROD_STEPS,
            "noe_csv": noe_dir / "noe_CYC2t.csv",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compute NOE r^-6 averages and cyclic phi/psi for CYC2 or CYC2t."
    )
    parser.add_argument(
        "--molecule",
        required=True,
        choices=["CYC2", "CYC2t"],
        help="Molecule to analyze.",
    )
    parser.add_argument(
        "--base-root",
        required=True,
        type=Path,
        help="Base directory containing simulation folders.",
    )
    parser.add_argument(
        "--noe-dir",
        required=True,
        type=Path,
        help="Directory containing NOE CSV files.",
    )
    parser.add_argument(
        "--stride", type=int, default=10, help="Stride used when loading trajectories."
    )
    parser.add_argument("--replica", type=int, default=0, help="REST2 replica index.")
    parser.add_argument(
        "--outdir", type=Path, default=Path("."), help="Output directory."
    )
    return parser


def main() -> int:
    """Run the CLI."""
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    systems = build_system_config(args.base_root, args.noe_dir)
    config = systems[args.molecule]

    sim_path = config["sim_path"]
    topology_pdb = sim_path / "npt" / "npt" / "init_topology.pdb"
    production_topology = topology_pdb

    noe_exp = load_noe_csv(config["noe_csv"])

    rest2_paths = build_rest2_paths(
        sim_path, replica=args.replica, n_steps=config["rest2_steps"]
    )
    traj_rest2 = load_segmented_trajectory(
        rest2_paths, top_file=topology_pdb, stride=args.stride
    )

    phi_psi_rest2 = phi_psi_df_cyclic(traj_rest2)
    phi_psi_rest2.to_csv(
        args.outdir / f"phi_psi_{args.molecule}_rest2.csv", index=False
    )

    noe_rest2, missing_rest2 = compute_noe_table(traj_rest2, noe_exp)
    noe_rest2.to_csv(
        args.outdir / f"NOE_results_{args.molecule}_rest2.csv", index=False
    )
    if len(missing_rest2):
        missing_rest2.to_csv(
            args.outdir / f"NOE_missing_{args.molecule}_rest2.csv", index=False
        )

    production_paths = build_production_paths(sim_path, n_steps=config["prod_steps"])
    traj_prod = load_segmented_trajectory(
        production_paths, top_file=production_topology, stride=args.stride
    )

    phi_psi_prod = phi_psi_df_cyclic(traj_prod)
    phi_psi_prod.to_csv(
        args.outdir / f"phi_psi_{args.molecule}_production.csv", index=False
    )

    noe_prod, missing_prod = compute_noe_table(traj_prod, noe_exp)
    noe_prod.to_csv(
        args.outdir / f"NOE_results_{args.molecule}_production.csv", index=False
    )
    if len(missing_prod):
        missing_prod.to_csv(
            args.outdir / f"NOE_missing_{args.molecule}_production.csv", index=False
        )

    print(
        f"[{args.molecule}] done. "
        f"REST2 frames={traj_rest2.n_frames}, "
        f"Production frames={traj_prod.n_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
