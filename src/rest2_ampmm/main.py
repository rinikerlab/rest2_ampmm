""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import argparse
import os

from mpi4py import MPI
from openmm import unit as u
from openmm.app import ForceField

from rest2_ampmm.helper_functions import (
    create_logger,
    get_num_gpus,
    load_yaml,
    read_molecules,
    set_constraint,
    set_integrator,
    set_nonbonded_method,
)
from rest2_ampmm.openmm_wrappers import (
    BarostatAdder,
    ForcefieldBuilder,
    PDBReader,
    ReporterAdder,
    SimulationBuilder,
    SimulationRunner,
    SolventAdder,
    SystemBuilder,
)
from rest2_ampmm.rest2 import (
    HarmonicRestraintsForREST2Adder,
    ParallelReplicaExchangeRunner,
)
from rest2_ampmm.torchforce import AmpConfigurator


def main():

    # Read-in YAML file from command-line input
    parser = argparse.ArgumentParser(
        description="This is a script that parses YAML file and performs the OpenMM simulation optionally with AMP-BMS model."
    )
    parser.add_argument(
        "parameters",
        type=str,
        help="Path to YAML file with all parameters required for a simulation.",
    )
    args = parser.parse_args()

    # Read all parameters from YAML file
    parameters = load_yaml(args.parameters)

    # If the parallel REST2 run is requested - get number of available GPUs
    if parameters["do_rest2"]:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        n_gpu = get_num_gpus()

    # Create simulation folder
    simulation_folder_path = parameters["base_path"]
    os.makedirs(simulation_folder_path, exist_ok=True)

    # Set logger
    if parameters["set_logger"] and (not parameters["do_rest2"]):
        logger = create_logger(os.path.join(simulation_folder_path, "allInfo.log"))

    elif parameters["set_logger"] and (parameters["do_rest2"]) and (rank == 0):
        logger = create_logger(
            os.path.join(simulation_folder_path, f"allInfo_{rank}.log")
        )

    else:
        logger = None

    # Create a forcefield for all molecules present in simulation
    forcefield = ForceField(*parameters["forcefield"]["default"])

    if parameters["molecules"]:
        molecules = read_molecules(parameters["molecules"])

    else:
        molecules = None

    if molecules:
        ff_builder = ForcefieldBuilder(forcefield=forcefield, logger=logger)

        ff_builder.parametrize_molecules_smirnoff(
            molecules=molecules,
            cache_path=parameters["cache_path"],
            ff_name=parameters["ff_name"],
        )

        forcefield = ff_builder.build_forcefield()

    else:
        if logger:
            message = "The molecules list is empty. Assume the all residues are known by the forcefield"
            logger.info(message)

    # Set the OpenMM integrator
    integrator = set_integrator(
        integrator_type=parameters["integrator_type"],
        integrator_parameters=parameters["integrator_parameters"],
    )

    # Read the PDB file
    pdb_reader = PDBReader(
        pdb_path=parameters["pdb_path"],
        restart=parameters["continue_simulation"],
        box_dimension=parameters["box_dimension"],
        solvate=parameters["solvate"],
        logger=logger,
    )

    # Get the OpenMM Modeller from the PDB file
    modeller = pdb_reader.get_modeller()

    # If solvation is requested - solvate the structure with OpenMM modeller.addSolvent method
    if parameters["solvate"] and not parameters["continue_simulation"]:
        solvent_adder = SolventAdder(modeller, forcefield, logger)

        solvent_adder.add_solvent(parameters["solvation_definition"])

    # create the OpenMM system
    system_builder = SystemBuilder(
        topology=modeller.topology,
        forcefield=forcefield,
        cutoff_nb=parameters["cutoff_nb"] * u.nanometer,
        nonbondedMethod=set_nonbonded_method(parameters["nonbondedMethod"]),
        rigidWater=parameters["rigidWater"],
        constraints=set_constraint(parameters["constraints"]),
        logger=logger,
    )

    system = system_builder.build_system()

    # If requested - add the barostat to the system
    if parameters["use_barostat"]:
        barostat_adder = BarostatAdder(
            system=system,
            barostat_type=parameters["barostat_type"],
            barostat_parameters=parameters["barostat_parameters"],
            logger=logger,
        )

        system = barostat_adder.modify_forces()

    # If REST2 run is requested, it is generally recommended to restraint bonds and angles of the system.
    # In replicas with significant scaling the bonds may dissociate without restraints
    if (
        parameters["do_rest2"]
        and parameters["use_AMP"]
        and parameters["restraint_bonds_angles_rest2"]
    ):
        restraint_adder = HarmonicRestraintsForREST2Adder(
            system=system,
            topology=modeller.topology,
            residue_names=parameters["qm_zone_resnames"],
            scale_factor_bond_rest2=parameters["state_parameters_definition"][rank][
                "scale_factor_bond_rest2"
            ],
            scale_factor_angle_rest2=parameters["state_parameters_definition"][rank][
                "scale_factor_angle_rest2"
            ],
            logger=logger,
        )

        restraint_adder.modify_forces()

    # Modify system to use AMP3 (do ML/MM with ML engine AMP-BMS)
    # The old forces also get affected, see logging and class description for further details
    if parameters["use_AMP"]:
        # Read the configuration parameters of AMP model
        parameters_ml = load_yaml(parameters["AMP_parameters_path"])

        if not parameters["do_rest2"] and not parameters["alchemical_FE"]:
            # Configure AMP model to be used in an unbiased MD simulation without alchemical transformations
            amp_configurator = AmpConfigurator(
                system=system,
                topology=modeller.topology,
                qm_zone_definition=parameters["qm_zone_resnames"],
                mm_zone_definition=parameters["mm_zone_resnames"],
                eps_rf=parameters_ml["eps_rf"],
                cutoff_nb=parameters["cutoff_nb"] * u.nanometer,
                params_path=parameters["AMP_parameters_path"],
                weights_path=parameters["weights_path"],
                device_ml=parameters["device_ml"],
                mol_charge=parameters["mol_charge"],
                scaling_charges=parameters["scaling_charges"],
                tip4p=parameters["tip4p"],
                logger=logger,
            )

        # Configure AMP model to be used in a REST2 MD simulation without alchemical transformations
        elif parameters["do_rest2"]:
            amp_configurator = AmpConfigurator(
                system=system,
                topology=modeller.topology,
                qm_zone_definition=parameters["qm_zone_resnames"],
                mm_zone_definition=parameters["mm_zone_resnames"],
                eps_rf=parameters_ml["eps_rf"],
                cutoff_nb=parameters["cutoff_nb"] * u.nanometer,
                params_path=parameters["AMP_parameters_path"],
                weights_path=parameters["weights_path"],
                device_ml=parameters["device_ml"],
                mol_charge=parameters["mol_charge"],
                scaling_factor_node_potential=parameters["state_parameters_definition"][
                    rank
                ]["scaling_factor_node_potential"],
                scaling_factor_coulomb_qm=parameters["state_parameters_definition"][
                    rank
                ]["scaling_factor_coulomb_qm"],
                scaling_factor_coulomb_qmmm=parameters["state_parameters_definition"][
                    rank
                ]["scaling_factor_coulomb_qmmm"],
                scaling_factor_D4=parameters["state_parameters_definition"][rank][
                    "scaling_factor_D4"
                ],
                scaling_factor_ZBL=parameters["state_parameters_definition"][rank][
                    "scaling_factor_ZBL"
                ],
                scaling_lj_qm_mm=parameters["state_parameters_definition"][rank][
                    "scaling_lj_qm_mm"
                ],
                scaling_charges=parameters["scaling_charges"],
                rank=int(rank % n_gpu),
                logger=logger,
            )

        # Configure AMP model to be used in an unbiased MD simulation with alchemical transformations
        elif parameters["alchemical_FE"]:
            amp_configurator = AmpConfigurator(
                system=system,
                topology=modeller.topology,
                qm_zone_definition=parameters["qm_zone_resnames"],
                mm_zone_definition=parameters["mm_zone_resnames"],
                eps_rf=parameters_ml["eps_rf"],
                cutoff_nb=parameters["cutoff_nb"] * u.nanometer,
                params_path=parameters["AMP_parameters_path"],
                weights_path=parameters["weights_path"],
                device_ml=parameters["device_ml"],
                mol_charge=parameters["mol_charge"],
                scaling_lj_qm_mm=parameters["alchemical_FE_definition"][
                    "scaling_factor_lj_qmmm"
                ],
                scaling_factor_alchemical_coulomb=parameters[
                    "alchemical_FE_definition"
                ]["scaling_factor_coulomb_qmmm"],
                scaling_charges=parameters["scaling_charges"],
                softcore_lj_qm_mm=True,
                logger=logger,
            )

        system = amp_configurator.configure()

    platform_propertis = None

    # Assign a force group based on the index
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)

    # Setup OpeMM simulation for REST2 run
    if parameters["do_rest2"]:
        platform_propertis = {}
        platform_propertis["DeviceIndex"] = str(rank % n_gpu)

        simulation_builder = SimulationBuilder(
            simulation_name=parameters["simulation_name"],
            forcefield=forcefield,
            integrator=integrator,
            system=system,
            modeller=modeller,
            platform_name=parameters["platform_name"],
            continue_simulation=parameters["continue_simulation"],
            state_xml_path=parameters["in_state_xml_path"],
            platform_properties=platform_propertis,
            do_rest2=parameters["do_rest2"],
            rank=rank,
            rest2_state_paths=parameters["rest2_state_paths"],
            logger=logger,
        )

    # Setup OpeMM simulation for unbiased MD with alchemical transformation
    elif parameters["alchemical_FE"]:
        simulation_builder = SimulationBuilder(
            simulation_name=parameters["simulation_name"],
            forcefield=forcefield,
            integrator=integrator,
            system=system,
            modeller=modeller,
            platform_name=parameters["platform_name"],
            continue_simulation=parameters["continue_simulation"],
            state_xml_path=parameters["in_state_xml_path"],
            platform_properties=platform_propertis,
            alchemical_FE=parameters["alchemical_FE"],
            scaling_lj_qm_mm=parameters["alchemical_FE_definition"][
                "scaling_factor_lj_qmmm"
            ],
            scaling_factor_alchemical_coulomb=parameters["alchemical_FE_definition"][
                "scaling_factor_coulomb_qmmm"
            ],
            logger=logger,
        )

    # Setup OpeMM simulation for unbiased MD without alchemical transformations involved
    else:
        simulation_builder = SimulationBuilder(
            simulation_name=parameters["simulation_name"],
            forcefield=forcefield,
            integrator=integrator,
            system=system,
            modeller=modeller,
            platform_name=parameters["platform_name"],
            continue_simulation=parameters["continue_simulation"],
            state_xml_path=parameters["in_state_xml_path"],
            platform_properties=platform_propertis,
            logger=logger,
        )

    simulation = simulation_builder.build_simulation()

    # Do energy minimization
    if parameters["minimize"] and not parameters["continue_simulation"]:
        if logger:
            message = """Minimizing the energy of the system..."""

            logger.info(message)

        simulation.minimizeEnergy(maxIterations=parameters["maxIterations"])

        if logger:
            message = """The system has been successfully minimized!!!"""

            logger.info(message)

    # Set the path to the folder where the initial topology and trajectory files will be saved
    production_folder_path = os.path.join(
        simulation_folder_path, parameters["simulation_name"]
    )

    # If needed - make this folder
    os.makedirs(production_folder_path, exist_ok=True)

    # Set the reporters for an unbiased MD (alchemical transformations are allowed)
    if not parameters["do_rest2"]:
        reporter_adder = ReporterAdder(
            simulation=simulation,
            out_folder_path=production_folder_path,
            readout_frequency=parameters["production_readout_frequency"],
            logger=logger,
        )

        # Save the current topology with coordinates after minimization to PDB file
        reporter_adder.write_initial_topology(
            file_name=parameters["initial_topology_name"]
        )

        # Add CSV reporter to track temperature/pressure/etc during the course of the simulation
        if parameters["add_prod_csv_reporter"]:
            simulation = reporter_adder.add_csv_reporter(
                file_name=parameters["production_csv_name"],
                parameters=parameters["production_csv_parameters"],
            )

        # Add HDF5 reporter to save the coordinates trajectory
        if parameters["add_prod_hdf5_reporter"]:
            simulation = reporter_adder.add_hdf5_reporter(
                file_name=parameters["production_hdf5_name"],
                parameters=parameters["production_hdf5_parameters"],
                residue_names_to_output=parameters["residue_names_to_output"],
                topology=simulation.topology,
            )

        # Add DCD reporter to save the coordinates trajectory
        if parameters["add_prod_dcd_reporter"]:
            simulation = reporter_adder.add_dcd_reporter(
                file_name=parameters["production_dcd_name"]
            )

    # Set the integrator step size
    simulation.integrator.setStepSize(parameters["integrator_parameters"]["step_size"])

    # Run the unbiased MD simulation (alchemical transformations allowed)
    if not parameters["do_rest2"]:
        simulation_runner = SimulationRunner(
            simulation=simulation,
            steps=parameters["production_steps"],
            run_type="production",
            save_final_state=parameters["save_final_state"],
            state_xml_path=parameters["out_state_xml_path"],
            logger=logger,
        )

        simulation_runner.run_simulation()

    # Configure the REST2 simulation. The reporters are added in the ParallelReplicaExchangeRunner
    # For further details see ParallelReplicaExchangeRunner implementation
    else:
        if logger:
            logger.info("REST2 run is possible to do only with AMP-based simulations!")

        replica_exchange_runner = ParallelReplicaExchangeRunner(
            comm=comm,
            rank=rank,
            parameters=parameters,
            system=system,
            simulation=simulation,
            out_folder_path=production_folder_path,
            logger=logger,
        )

        replica_exchange_runner.run_simulation()


if __name__ == "__main__":
    main()
