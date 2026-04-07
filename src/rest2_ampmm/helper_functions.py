""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Set, Tuple, Union

import numpy as np
import openmm as mm
import openmm.app as app
import yaml
from openff.toolkit import Molecule
from openmm import unit as u
from pycuda import driver

from rest2_ampmm.utils import Constraints, NonbondedMethod


def get_num_gpus() -> int:
    """Returns the number of GPUs available"""
    try:
        driver.init()
        num_gpus = driver.Device.count()
        return num_gpus
    except:
        return 0


def load_yaml(path: str) -> Dict[str, Any]:
    """Function to read YAML file and convert to python dictionary

    Args:
        path (str): Path to the YAML file

    Returns:
        Dict[str, Any]: Dict containing YAML data
    """
    parameters = yaml.safe_load(Path(path).read_text())

    return parameters


def create_logger(path: str) -> logging.Logger:
    """Function to get a pre-configured python logger
    Args:
        path (str): Path to the file to which the logger will write its output

    Returns:
        logging.Logger: Logger instance
    """

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(path)

    file_handler.setLevel(logging.DEBUG)

    f_format = logging.Formatter("%(asctime)s - %(message)s")

    file_handler.setFormatter(f_format)

    logger.addHandler(file_handler)

    return logger


def read_molecules(mol_definitions: Iterable[str]) -> List[Molecule]:
    """Function to convert an iterable of SDF files or SMILES to list of OpenFF Molecule objects

    Args:
        mol_definitions (Iterable[str]): Collection of SDF file paths or SMILES strings

    Returns:
        List[Molecule]: List of OpenFF Molecule objects created from given input
    """

    molecules = []

    for mol_definition in mol_definitions:
        if mol_definition.endswith(".sdf"):
            molecule = Molecule(mol_definition)

        else:
            molecule = Molecule.from_smiles(mol_definition)

        molecules.append(molecule)

    return molecules


def set_integrator(
    integrator_type: Literal["LMI"], integrator_parameters: Dict[str, float]
) -> mm.openmm.Integrator:
    """Function that configures the OpenMM integrator.

    Args:
        integrator_type (Literal[&quot;LMI&quot;]): OpenMM integrator abbreviation.
        "LMI" returns LangevinMiddleIntegrator
        integrator_parameters (Dict[str, float]): Dictionary with keys matching the OpenMM parameter names of the respective integrator
        temperature in [K], friction_coefficient in [1/picoseconds], step_size [picoseconds]

    Raises:
        ValueError: If the unsupported key is given.

    Returns:
        mm.openmm.Integrator: OpenMM Integrator with set temperature/friction_coefficient/step_size
    """
    supported_keys = set(["LMI"])

    if integrator_type not in supported_keys:
        raise ValueError(
            "The function supports right now only LMI - LangevinMiddleIntegrator"
        )

    match integrator_type:
        case "LMI":
            temp = integrator_parameters["temperature"] * u.kelvin
            fric_coeff = integrator_parameters["friction_coefficient"] / u.picosecond
            step_size = integrator_parameters["step_size"] * u.picoseconds

            integrator = mm.openmm.LangevinMiddleIntegrator(temp, fric_coeff, step_size)

    return integrator


def set_constraint(constraint_type: Literal["HBonds", ""]) -> Constraints:
    """Convert string into Constraints enum.

    ""  -> no constraints
    """

    if constraint_type == "":
        return Constraints.None_

    return Constraints[constraint_type]


def set_nonbonded_method(
    nonbonded_method_type: Literal[
        "PME", "NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "LJPME"
    ],
) -> NonbondedMethod:
    """Convert string into NonbondedMethod enum."""

    return NonbondedMethod[nonbonded_method_type]


def get_atom_ids(
    residue_names: Iterable[str], topology: app.topology.Topology
) -> Dict[str, list[int]]:
    """Function to get the atomic IDs (starts counting from 1, not from 0 as are indices in OpenMM)

    Args:
        residue_names (Iterable[str]): Collection of residue names
        topology (app.topology.Topology): Valid OpenMM Topology

    Returns:
        Dict[str, list[int]]: Atomic IDs by residue name
    """
    resnames_in_topology = set(res.name for res in topology.residues())

    if not set(residue_names).issubset(resnames_in_topology):
        raise ValueError(
            "Some of the passsed residue names are not present in pdbfile!"
        )

    res_names = set(residue_names)

    output = dict()

    for name in res_names:
        output[name] = []

    for atom in topology.atoms():
        if atom.residue.name in res_names:
            output[atom.residue.name].append(int(atom.id))

    return output


def get_atom_indices(
    residue_names: Iterable[str], topology: app.topology.Topology
) -> Dict[str, list[int]]:
    """Function to get the atomic indices (0-based) of the specified residue names

    Args:
        residue_names (Iterable[str]): Collection of residue names for which the indices should be found
        topology (_type_): Valid OpenMM Topology

    Returns:
        Dict[str, list[int]]: Atomic indices by residue name
    """

    resnames_in_topology = set(res.name for res in topology.residues())

    if not set(residue_names).issubset(resnames_in_topology):
        raise ValueError("Some of the passed residue names are not present in pdbfile!")

    res_names = set(residue_names)

    output = dict()

    for name in res_names:
        output[name] = []

    for atom in topology.atoms():
        if atom.residue.name in res_names:
            output[atom.residue.name].append(int(atom.index))

    return output


def write_qm_mm_json(
    path: str, qm_zone_resnames: List[str], mm_zone_resnames: List[str]
) -> None:
    """Function to write QM/MM JSON file

    Args:
        path (str): Path to the JSON file (with file extension), where the QM/MM zones will be defined
        qm_zone_resnames (List[str]): Collection of QM zone residue names
        mm_zone_resnames (List[str]): Collection of MM zone residue names

    """
    parameters = {}
    parameters["qm_zone_resnames"] = qm_zone_resnames
    parameters["mm_zone_resnames"] = mm_zone_resnames

    with open(path, "w") as file:
        json.dump(parameters, file, indent=4)


def write_solvation_json(
    path: str,
    water_model: Literal["TIP3P", "TIP4P-FB"] = "TIP4P-FB",
    boxSize: Union[List[float], Tuple[float], None] = None,
    boxVectors: Union[List[List[float]], Tuple[Tuple[float]], None] = None,
    padding: Union[float, None] = None,
    numAdded: Union[int, None] = None,
    neutralize: bool = False,
) -> None:
    """Function to write the solvation JSON file.

    Args:
        path (str):  Path to the JSON file (with file extension), where all data needed for solvation definition is stored
        water_model (Literal[&quot;TIP3P&quot;, &quot;TIP4P, optional): Water model to be used for solvation. Defaults to "TIP4P-FB".
        boxSize (Union[List[float], Tuple[float], None], optional): Dimensions of the box after solvation, in [nm]. Defaults to None.
        boxVectors (Union[List[List[float]], Tuple[Tuple[float]], None], optional): Box vectors after solvation, in [nm]. Defaults to None.
        padding (Union[float, None], optional): Padding to be used for solvation, in [nm]. Defaults to None.
        numAdded (Union[int, None], optional): Number of water molecules to be used for solvation. Defaults to None.
        neutralize (bool, optional): Set to True if you want to add ions to neutralize the box. Defaults to False.
    """
    parameters = {}
    parameters["water_model"] = water_model
    parameters["boxSize"] = boxSize
    parameters["boxVectors"] = boxVectors
    parameters["padding"] = padding
    parameters["numAdded"] = numAdded
    parameters["neutralize"] = neutralize

    with open(path, "w") as file:
        json.dump(parameters, file, indent=4)


def dictoflists2set(dictionary: Dict[str, List[int]]) -> Set[int]:
    """Helper function to unite all values of a dictionary (which are lists) in one set

    Args:
        dictionary (Dict[str, List[int]]): Input dictionary

    Returns:
        set: Resulting set
    """
    result_set = set()
    for sublist in dictionary.values():
        result_set.update(sublist)
    return result_set


def read_molecules_file(path: str) -> List[str]:
    """Helper function to parse the text file and convert it into list of lines

    Args:
        path (str): Path to the file

    Returns:
        List[str]: Result list
    """

    if path is None:
        return None

    with open(path, "r") as file:
        all_lines = file.readlines()

    return [line.strip() for line in all_lines if line.strip() != ""]


def jsonfile2dict(path: str) -> Dict:
    """Helper function to read the JSON file as python dictionary

    Args:
        path (str): Path to the JSON file

    Returns:
        Dict: Result dictionary
    """
    with open(path, "r") as file:
        return json.load(file)


def readjsonfile(path: str):

    with open(path, "r") as file:
        data = json.load(file)

    return data


def write_rest2_states_json(
    path: str,
    num_states: int,
    scaled_params: List[str],
    upper_bound: float = 0.125,
    lower_bound: float = 1.0,
) -> None:
    """Helper function to write the JSON file with scaling parameters for all states of REST2

    Args:
        path (str): Path to the JSON file (with file extension), where all scaling parameters for all states will be stored
        num_states (int): Number of states in REST2
        scaled_params (List[str]): Collection of names of REST2 parameters that are active in REST2 scaling
        upper_bound (float, optional): Maximum scaling of REST2 parameter. Defaults to 0.125.
        lower_bound (float, optional): Minimum scaling of REST2 parameter. Defaults to 1.0.
    """

    scaling_factors = np.linspace(lower_bound, upper_bound, num_states)
    reversed_scaling_factors = np.linspace(upper_bound, lower_bound, num_states - 1)

    basis_state = {
        "scaling_factor_node_potential": 1.0,
        "scaling_factor_coulomb_qm": 1.0,
        "scaling_factor_coulomb_qmmm": 1.0,
        "scaling_factor_D4": 1.0,
        "scaling_factor_ZBL": 1.0,
        "scaling_lj_qm_mm": 1.0,
        "scale_factor_bond_rest2": 0.0,
        "scale_factor_angle_rest2": 0.0,
    }

    parameter_states = []

    for i, scaling_factor in enumerate(scaling_factors):
        state = deepcopy(basis_state)

        for scaled_param in scaled_params:
            if (
                scaled_param == "scaling_factor_coulomb_qmmm"
                or scaled_param == "scaling_lj_qm_mm"
            ):
                scaling_factor_sqrt = np.sqrt(scaling_factor)
                state[scaled_param] = scaling_factor_sqrt
                continue

            elif (
                scaled_param == "scale_factor_bond_rest2"
                or scaled_param == "scale_factor_angle_rest2"
            ):
                if i == 0:
                    state[scaled_param] = 0

                else:
                    state[scaled_param] = reversed_scaling_factors[i - 1]

                continue

            state[scaled_param] = scaling_factor

        parameter_states.append(state)

    with open(path, "w") as file:
        json.dump(parameter_states, file, indent=4)


def write_first_rest2_xml_paths_json(
    save_path: str, base_path: str, num_states: int, file_name: str
) -> None:
    """Helper function to write the JSON file with XML state file paths for REST2 run continue/restart.
    Every REST2 replica will be restarted/started from the same XML file.

    Args:
        save_path (str): Path to the JSON file (with file extension), where all XML state paths are stored
        base_path (str): Path to the folder, where the XML restart file is located
        num_states (int): Number of REST2 replicas (states)
        file_name (str): Name of the restart XML file (no extension)
    """

    state_paths = []

    for _ in range(num_states):
        state_paths.append(os.path.join(base_path, f"{file_name}.xml"))

    with open(save_path, "w") as file:
        json.dump(state_paths, file, indent=4)


def write_rest2_xml_paths_json(
    save_path: str, base_path: str, num_states: int, file_name: str
) -> None:
    """Helper function to write the JSON file with OpenMM state XML file paths that should be used for
    restart/continuation of all replicas in REST2-AMP/MM run.

    Args:
        save_path (str): Path to the JSON file (with file extension), where all XML checkpoint paths are stored
        base_path (str): Path to the folder, where the XML restart files are located
        num_states (int): Number of REST2 replicas (states)
        file_name (str): Name of the restart XML file (no extension)
    """
    state_paths = []

    for i in range(num_states):
        state_paths.append(os.path.join(base_path, f"{file_name}_{i:04d}.xml"))

    with open(save_path, "w") as file:
        json.dump(state_paths, file, indent=4)


def generate_yaml_config(
    file_path: str,
    base_path: str,
    simulation_name: str,
    pdb_path: str,
    mol_charge: int,
    forcefield_path: str,
    **kwargs,
):
    # Required fields
    all_parameters = {
        "base_path": base_path,
        "simulation_name": simulation_name,
        "pdb_path": pdb_path,
        "mol_charge": mol_charge,
        "molecules": read_molecules_file(kwargs.get("molecules_file", None)),
        "forcefield": jsonfile2dict(forcefield_path),
        "solvate": kwargs.get("solvate", True),
        "tip4p": kwargs.get("tip4p", True),
        "set_logger": kwargs.get("set_logger", True),
        "cache_path": kwargs.get("cache_path", None),
        "ff_name": kwargs.get("ff_name", "openff_unconstrained-2.2.0"),
        "integrator_type": kwargs.get("integrator_type", "LMI"),
        "integrator_parameters": {
            "temperature": kwargs.get("temperature", 298.15),
            "friction_coefficient": kwargs.get("integrator_fric_coeff", 1.0),
            "step_size": kwargs.get("step_size", 0.0005),
        },
        "use_barostat": kwargs.get("use_barostat", False),
        "set_to_temperature": kwargs.get("set_to_temperature", False),
        "minimize": kwargs.get("minimize", True),
        "maxIterations": kwargs.get("max_iterations", 0),
        "box_dimension": kwargs.get("box_dimension", None),
        "cutoff_nb": kwargs.get("cutoff_nb", 0.9),
        "nonbondedMethod": kwargs.get("nonbondedMethod", "PME"),
        "rigidWater": kwargs.get("rigidWater", True),
        "constraints": kwargs.get("constraints", ""),
        "platform_name": kwargs.get("platform_name", "CUDA"),
        "continue_simulation": kwargs.get("continue_simulation", False),
        "in_state_xml_path": kwargs.get("in_state_xml_path", None),
        "save_final_state": kwargs.get("save_final_state", False),
        "initial_topology_name": kwargs.get(
            "initial_topology_name", "init_topology.pdb"
        ),
        "production_readout_frequency": kwargs.get(
            "production_readout_frequency", 1000
        ),
        "production_steps": kwargs.get("production_steps", 500000),
        "add_prod_csv_reporter": kwargs.get("add_prod_csv_reporter", False),
        "add_prod_hdf5_reporter": kwargs.get("add_prod_hdf5_reporter", False),
        "add_prod_dcd_reporter": kwargs.get("add_prod_dcd_reporter", False),
        "use_AMP": kwargs.get("use_AMP", False),
        "scaling_charges": kwargs.get("scaling_charges", False),
        "do_rest2": kwargs.get("do_rest2", False),
        "alchemical_FE": kwargs.get("alchemical_FE", False),
        "numrep": kwargs.get("numrep", 1),
    }

    # Optional extras

    if all_parameters["use_barostat"]:
        all_parameters["barostat_type"] = kwargs.get("barostat_type", "MCB")
        all_parameters["barostat_parameters"] = {
            "pressure": kwargs.get("barostat_pressure", 1.0),
            "temperature": kwargs.get("barostat_temperature", 298.15),
            "frequency": kwargs.get("barostat_frequency", 25),
        }

    if all_parameters["solvate"]:
        all_parameters["solvation_definition"] = readjsonfile(
            kwargs.get("solvation_definition")
        )

    if all_parameters["save_final_state"] and kwargs.get("out_state_xml_path"):
        all_parameters["out_state_xml_path"] = kwargs.get("out_state_xml_path")

    if all_parameters["add_prod_csv_reporter"]:
        all_parameters["production_csv_name"] = kwargs.get(
            "prod_csv_name", "production_properties_trajectory.csv"
        )
        all_parameters["production_csv_parameters"] = jsonfile2dict(
            kwargs.get("prod_csv_parameters")
        )

    if all_parameters["add_prod_hdf5_reporter"]:
        all_parameters["production_hdf5_name"] = kwargs.get(
            "prod_hdf5_name", "production_trajectory.h5"
        )
        all_parameters["production_hdf5_parameters"] = jsonfile2dict(
            kwargs.get("prod_hdf5_parameters")
        )
        all_parameters["residue_names_to_output"] = kwargs.get(
            "residue_names_to_output"
        )

    if all_parameters["add_prod_dcd_reporter"]:
        all_parameters["production_dcd_name"] = kwargs.get(
            "prod_dcd_name", "production_trajectory.dcd"
        )

    if all_parameters["use_AMP"]:
        all_parameters["AMP_parameters_path"] = kwargs.get("AMP_parameters_path")
        all_parameters["weights_path"] = kwargs.get("weights_path")
        all_parameters["device_ml"] = kwargs.get("device_ml", "cuda")
        tmp = jsonfile2dict(kwargs.get("qm_mm_zones_definition"))
        all_parameters["qm_zone_resnames"] = tmp["qm_zone_resnames"]
        all_parameters["mm_zone_resnames"] = tmp["mm_zone_resnames"]

    all_parameters["rest2_parameter_names"] = [
        "scaling_factor_node_potential",
        "scaling_factor_coulomb_qm",
        "scaling_factor_coulomb_qmmm",
        "scaling_factor_D4",
        "scaling_factor_ZBL",
        "scaling_lj_qm_mm",
    ]

    if all_parameters["do_rest2"]:
        all_parameters["exchange_frequency"] = kwargs.get("exchange_frequency")
        all_parameters["state_parameters_definition"] = readjsonfile(
            kwargs.get("state_parameters_definition")
        )
        all_parameters["exchange_stats_path"] = kwargs.get("exchange_stats_path")

        if all_parameters["continue_simulation"]:
            all_parameters["rest2_state_paths"] = readjsonfile(
                kwargs.get("rest2_state_paths")
            )

        all_parameters["rest2_save_checkpoints"] = kwargs.get(
            "rest2_save_checkpoints", True
        )

        all_parameters["restraint_bonds_angles_rest2"] = kwargs.get(
            "restraint_bonds_angles_rest2", True
        )

        if all_parameters["restraint_bonds_angles_rest2"]:
            #####Adding New Replica Exchange Parameters#########
            all_parameters["rest2_parameter_names"].extend(
                ["scale_factor_bond_rest2", "scale_factor_angle_rest2"]
            )

    if all_parameters["alchemical_FE"]:
        all_parameters["alchemical_FE_definition"] = readjsonfile(
            kwargs.get("alchemical_FE_definition")
        )

    # Write to YAML
    with open(file_path, "w") as yaml_file:
        yaml.dump(all_parameters, yaml_file, indent=4, sort_keys=True)
