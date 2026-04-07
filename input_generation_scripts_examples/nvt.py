""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
from pathlib import Path

from rest2_ampmm.helper_functions import (
    generate_yaml_config,
    write_qm_mm_json,
    write_solvation_json,
)


def main():
    # =========================
    # I/O
    # =========================
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--molecule", type=str, help="Name of the molecules (e.g. ALA, CYC, TRP)"
    )
    parser.add_argument(
        "--mol_charge", type=int, default=0, help="Charge of the molecule"
    )
    parser.add_argument("--qm_zone", nargs="+", help="QM residues")
    parser.add_argument("--mm_zone", nargs="+", help="MM residues")
    parser.add_argument(
        "--input", type=str, help="Name of the input (previous simulation phase)"
    )
    parser.add_argument(
        "--output", type=str, help="Name of the output (current simulation phase)"
    )

    args = parser.parse_args()

    # =========================
    # USER SETTINGS (edit here)
    # =========================
    molecule = args.molecule
    simulation_phase = args.output
    mol_charge = args.mol_charge
    molecules_file = None

    # Zones
    qm_zone_residues = args.qm_zone
    mm_zone_residues = args.mm_zone

    # Solvation
    box_dimension = 4.0
    padding = 1.5  # nm of solvent padding around solute
    tip4p = True

    # Thermo / Integrator
    temp = 298.15  # K
    friction = 1.0  # 1/ps
    step_size = 0.001  # ps
    use_barostat = False
    set_to_temperature = False
    minimize = True
    max_iterations = 0  # 0 = until convergence

    # Nonbonded / Constraints
    cutoff_nb = 0.9  # nm
    nonbonded_method = "PME"  # Will switch to RF if AMP is used
    rigid_water = True
    constraints = ""

    # Platform
    platform_name = "CUDA"
    device_ml = "cuda"  # for AMP

    # Production
    production_steps = 100000  # 100 ps
    prod_read_freq = 1000
    continue_simulation = False
    save_final_state = True

    # Reporters
    add_prod_csv_reporter = True
    add_prod_hdf5_reporter = True
    add_prod_dcd_reporter = False

    # AMP / QM-MM / Advanced
    use_AMP = True
    scaling_charges = 1.0
    do_rest2 = False
    alchemical_FE = False
    numrep = 1

    # =========================
    # PROJECT PATHS
    # =========================
    root_dir = Path(__file__).resolve().parent.parent

    # Data & simulation roots
    resources_dir = root_dir / "resources"
    data_dir = root_dir / "data"
    default_configs_dir = root_dir / "configs" / "default"
    amp_root = root_dir / "amp_simulation" / molecule

    # Input files
    pdb_path = data_dir / "starting_structures" / f"{molecule}.pdb"
    amp_parameters_path = (
        resources_dir / "parameters" / "PARAMETERS_MIN_LRv2_tip4pfb.yaml"
    )
    weights_path = resources_dir / "weights" / "MIN_LRv2_state_dict"
    forcefield_path = default_configs_dir / "ff_default_tip4pfb.json"
    csv_json_path = default_configs_dir / "default_csv_parameters.json"
    hdf_json_path = default_configs_dir / "default_hdf5_parameters.json"

    # Phase-specific folder & outputs
    simulation_folder = amp_root / simulation_phase
    simulation_folder.mkdir(parents=True, exist_ok=True)
    xml_save_path = simulation_folder / f"{simulation_phase}.xml"
    yaml_path = simulation_folder / "config.yaml"
    qm_mm_zone_json_path = simulation_folder / "qm_mm_zone_definition.json"
    solvation_parameters_path = simulation_folder / "solvation.json"

    # =========================
    # PREP INPUT DEFINITIONS
    # =========================
    # Zones definition
    write_qm_mm_json(qm_mm_zone_json_path, qm_zone_residues, mm_zone_residues)

    # Solvation definition
    # (If write_solvation_json supports a path, pass it; otherwise it writes where it expects.)
    write_solvation_json(solvation_parameters_path, padding=padding)

    # =========================
    # BUILD CONFIG
    # =========================
    config = {
        # ---- I/O & bookkeeping ----
        "file_path": str(yaml_path),
        "base_path": str(simulation_folder),
        "simulation_name": str(simulation_phase),
        "pdb_path": str(pdb_path),
        "initial_topology_name": "init_topology.pdb",
        "set_logger": True,
        "cache_path": None,
        "ff_name": None,
        "forcefield_path": str(forcefield_path),
        "out_state_xml_path": str(xml_save_path),
        "save_final_state": save_final_state,
        "continue_simulation": continue_simulation,
        # ---- System composition ----
        "mol_charge": mol_charge,
        "molecules_file": molecules_file,
        "solvate": True,
        "solvation_definition": str(solvation_parameters_path),
        "tip4p": tip4p,
        "box_dimension": box_dimension,
        # ---- Thermostat / Integrator ----
        "integrator_type": "LMI",  # Langevin Middle Integrator
        "temperature": temp,
        "friction_coefficient": friction,
        "step_size": step_size,
        "use_barostat": use_barostat,
        "set_to_temperature": set_to_temperature,
        "minimize": minimize,
        "maxIterations": max_iterations,
        # ---- Nonbonded / constraints ----
        "cutoff_nb": cutoff_nb,
        "nonbondedMethod": nonbonded_method,  # switched to RF if AMP is used
        "rigidWater": rigid_water,
        "constraints": constraints,
        # ---- Platform ----
        "platform_name": platform_name,
        # ---- Production control ----
        "production_steps": production_steps,
        "production_readout_frequency": prod_read_freq,
        # ---- Reporters ----
        "add_prod_csv_reporter": add_prod_csv_reporter,
        "prod_csv_parameters": csv_json_path,
        "add_prod_hdf5_reporter": add_prod_hdf5_reporter,
        "prod_hdf5_parameters": hdf_json_path,
        "residue_names_to_output": qm_zone_residues + mm_zone_residues,
        "add_prod_dcd_reporter": add_prod_dcd_reporter,
        # ---- AMP settings ----
        "use_AMP": use_AMP,
        "AMP_parameters_path": str(amp_parameters_path),
        "weights_path": str(weights_path),
        "device_ml": device_ml,
        # ---- QM/MM & advanced options ----
        "qm_mm_zones_definition": str(qm_mm_zone_json_path),
        "scaling_charges": scaling_charges,
        "do_rest2": do_rest2,
        "alchemical_FE": alchemical_FE,
        "numrep": numrep,
    }

    # Write the YAML configuration file
    generate_yaml_config(**config)

    print(f"\nConfig created at:\n{yaml_path}")


if __name__ == "__main__":
    main()
