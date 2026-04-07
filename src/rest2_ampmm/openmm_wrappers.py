""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import os
import time
from logging import Logger
from typing import Any, Dict, Iterable, List, Literal, Tuple, Union

import mdtraj as md
import openmm as mm
from openff.toolkit import Molecule
from openmm import unit as u
from openmm.app import ForceField, Modeller, PDBFile, Simulation
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from rest2_ampmm.helper_functions import get_atom_indices
from rest2_ampmm.utils import Constraints, ForcesModifier, NonbondedMethod


class SolventAdder:
    """Class that is used to add solvent molecules to the OpenMM modeller.
    Is just a wrapper of the OpenMM modeller.addSolvent method.
    """

    def __init__(
        self,
        modeller: Modeller,
        forcefield: ForceField,
        logger: Union[None, Logger] = None,
    ) -> None:
        """Constructor

        Args:
            modeller (Modeller): Valid OpenMM Modeller
            forcefield (ForceField): Valid OpenMM ForceField
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.modeller = modeller

        self.forcefield = forcefield

        self.logger = logger

    def _translate_water_model(self, water_model: Literal["TIP3P", "TIP4P-FB"]) -> str:
        """Helper function to translate from standard names of water models to water model names of OpenMM.

        Args:
            water_model (Literal[&quot;TIP3P&quot;, &quot;TIP4P): Supported water models.

        Returns:
            str: name of OpenMM water model.
        """

        translation = {
            "TIP3P": "tip3p",
            "TIP4P-FB": "tip4pew",
        }

        return translation[water_model]

    def _convert_to_vec3(self, value: Union[List[float], Tuple[float]]) -> mm.vec3.Vec3:
        """Helper function to get the mm.vec3.Vec3 object from a list/tupleof 3 floats.
        The numbers are assumed to be in [nm].

        Args:
            value (Union[List[float], Tuple[float]]): Numbers in [nm]

        Raises:
            ValueError: If the given iterable provides not 3 floats, it cannot be converted to mm.vec3.Vec3 object.

        Returns:
            mm.vec3.Vec3: OpenMM vector in [nm]
        """

        if isinstance(value, (list, tuple)) and len(value) == 3:
            # Convert to Vec3 assuming value is in Nanometer
            return mm.vec3.Vec3(*[val * u.nanometer for val in value])

        raise ValueError(f"Cannot convert {value} to Vec3")

    def _get_boxSize(
        self, boxSize: Union[List[float], Tuple[float]]
    ) -> Union[mm.vec3.Vec3, None]:
        """Helper function to get the boxSize parameter in [nm] for modeller.addSolvent method.

        Args:
            boxSize (Union[List[float], Tuple[float]]): Box dimensions in [nm]

        Returns:
            Union[mm.vec3.Vec3, None]: OpenMM vector of box dimensions in [nm]
        """

        if boxSize is not None:
            return self._convert_to_vec3(boxSize)

        else:
            return None

    def _get_boxVectors(
        self, boxVectors: Union[List[List[float]], Tuple[Tuple[float]]]
    ) -> Union[None, Tuple[mm.vec3.Vec3, mm.vec3.Vec3, mm.vec3.Vec3]]:
        """Helper function to get boxVectors parameter in [nm] for modeller.addSolvent method.

        Args:
            boxVectors (Union[List[List[float]], Tuple[Tuple[float]]]): Box vectors in [nm]

        Raises:
            ValueError: If not 3 box vectors are provided, the box cannot be defined.

        Returns:
            Union[None, Tuple[mm.vec3.Vec3, mm.vec3.Vec3, mm.vec3.Vec3]]: Box vectors as OpenMM vectors in [nm]
        """

        if boxVectors is not None:
            if len(boxVectors) != 3:
                raise ValueError("boxVectors must be a tuple/list of 3 vectors")

            return tuple(self._convert_to_vec3(vec) for vec in boxVectors)

        else:
            return None

    def _get_padding(self, padding: Union[int, float]) -> mm.unit.quantity.Quantity:
        """Helper function to get the padding parameter in [nm] for modeller.addSolvent method

        Args:
            padding (Union[int, float]): Padding value in [nm]

        Raises:
            TypeError: Padding must be either int or float!

        Returns:
            _type_: Padding as mm.unit.quantity.Quantity
        """

        if padding is not None:
            if isinstance(padding, (int, float)):
                return padding * u.nanometer

            else:
                raise TypeError("padding must be either int or float!")
        else:
            return None

    def add_solvent(self, solvation_definition: Dict[str, Any]) -> None:
        """Function that is used to add solvent with openmm modeller.addSolvent method based on configuration data.

        Args:
            solvation_definition (Dict[str, Any]): Configuration data for adding solvent with OpenMM modeller.
        """

        self.modeller.addSolvent(
            forcefield=self.forcefield,
            model=self._translate_water_model(solvation_definition["water_model"]),
            boxSize=self._get_boxSize(solvation_definition["boxSize"]),
            boxVectors=self._get_boxVectors(solvation_definition["boxVectors"]),
            padding=self._get_padding(solvation_definition["padding"]),
            numAdded=solvation_definition["numAdded"],
            neutralize=solvation_definition["neutralize"],
        )

        self.modeller.addExtraParticles(self.forcefield)

        if self.logger:
            pass


class PDBReader:
    def __init__(
        self,
        pdb_path: str,
        solvate: bool,
        restart: bool = False,
        box_dimension: Union[None, float] = None,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            pdb_path (str): Path to the PDF file
            solvate (bool): Set to True if the addition of solvent if required.
            restart (bool, optional): Set to True if restart from the checkpoint is reauired. Defaults to False.
            box_dimension (Union[None, float], optional): Size of the box edge in [nm]. Defaults to None.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.pdb_path = pdb_path

        self.solvate = solvate

        self.restart = restart

        if not solvate and not restart and box_dimension is None:
            raise ValueError(
                "box_dimension should be a float if solvate==False (the box is treated complete, so the box vectors should be specified)."
            )

        self.box_dimension = box_dimension

        self.logger = logger

    # Helper function to read the PDB file and optionally log what has been read
    def _read_pdb(self) -> None:
        """Helper function to read-in PDB file with OpenMM PDBFile and optionally log the read information"""

        self.pdbfile = PDBFile(self.pdb_path)

        if self.logger:
            message = f"""
            The topology and atom positions are read from:  {self.pdb_path}\n
            Total number of read atoms:     {self.pdbfile.topology.getNumAtoms()}\n
            Total number of read positions:     {len(self.pdbfile.getPositions())}\n
            Total number of read residues:      {self.pdbfile.topology.getNumResidues()}\n
            """

            self.logger.info(message)

    def _prepare_modeller_from_pdbfile(self) -> None:
        """Helper function to get the OpenMM Modeller from PDBFile, and set PeriodicBoxVectors if necessary"""

        self.modeller = Modeller(self.pdbfile.topology, self.pdbfile.positions)

        if not self.solvate and not self.restart:
            a = mm.vec3.Vec3(
                x=self.box_dimension, y=self.box_dimension * 0, z=self.box_dimension * 0
            )
            b = mm.vec3.Vec3(
                x=self.box_dimension * 0, y=self.box_dimension, z=self.box_dimension * 0
            )
            c = mm.vec3.Vec3(
                x=self.box_dimension * 0, y=self.box_dimension * 0, z=self.box_dimension
            )

            box_vectors = (a, b, c)

            self.modeller.topology.setPeriodicBoxVectors(box_vectors)

            box_vectors_with_units = self.modeller.topology.getPeriodicBoxVectors()

            if self.logger:
                self.logger.info(
                    f"The box dimensions are set to: \t a={box_vectors_with_units[0]} \t b={box_vectors_with_units[1]} \t c={box_vectors_with_units[2]}"
                )

        elif self.restart:
            mock_box_dim = 10.0

            a = mm.vec3.Vec3(x=mock_box_dim, y=mock_box_dim * 0, z=mock_box_dim * 0)
            b = mm.vec3.Vec3(x=mock_box_dim * 0, y=mock_box_dim, z=mock_box_dim * 0)
            c = mm.vec3.Vec3(x=mock_box_dim * 0, y=mock_box_dim * 0, z=mock_box_dim)

            box_vectors = (a, b, c)

            self.modeller.topology.setPeriodicBoxVectors(box_vectors)

            box_vectors_with_units = self.modeller.topology.getPeriodicBoxVectors()

            if self.logger:
                self.logger.info(
                    f"""Restart of the simulation has been requested!\nAdded mock box: \t a={box_vectors_with_units[0]} \t b={box_vectors_with_units[1]} \t c={box_vectors_with_units[2]}\nThey will be overwritten by box dimensions from the restart XML file"""
                )

    def get_modeller(self):
        """Get the OpenMM Modeller object from PDB file path"""

        self._read_pdb()

        self._prepare_modeller_from_pdbfile()

        return self.modeller


class SystemBuilder:
    def __init__(
        self,
        topology: mm.app.topology.Topology,
        forcefield: mm.app.forcefield.ForceField,
        cutoff_nb: mm.unit.quantity.Quantity,
        nonbondedMethod: NonbondedMethod = NonbondedMethod.PME,
        rigidWater: bool = True,
        constraints: Constraints = Constraints.None_,
        logger: Union[None, Logger] = None,
    ) -> None:
        """Constructor

        Args:
            topology (mm.app.topology.Topology): Valid OpenMM Topology
            forcefield (mm.app.forcefield.ForceField): Valid OpenMM ForceField
            cutoff_nb (mm.unit.quantity.Quantity): Cutoff nonbonded, in [nm]
            nonbondedMethod (NonbondedMethod, optional): Nonbonded Method that will be overwritten by AMP,
            but from which the custom RF will be built. Defaults to NonbondedMethod.PME.
            rigidWater (bool, optional): Set to True if using rigid water models. Defaults to True.
            constraints (Constraints, optional): Set to HBonds if using classical FF. Set to None if using AMP. Defaults to Constraints.None_.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.topology = topology
        self.forcefield = forcefield
        self.cutoff_nb = cutoff_nb
        self.nonbondedMethod = nonbondedMethod.value
        self.rigidWater = rigidWater
        self.constraints = constraints.value
        self.logger = logger

    def build_system(self):
        """Function that builds the OpenMM system using provided arguments with optional logging"""

        system = self.forcefield.createSystem(
            self.topology,
            nonbondedMethod=self.nonbondedMethod,
            nonbondedCutoff=self.cutoff_nb,
            rigidWater=self.rigidWater,
            constraints=self.constraints,
        )

        if self.logger:
            message = f"""The system is created with the following settings:  
            Nonbonded Method:   {self.nonbondedMethod}\n
            Nonbonded Cutoff:   {self.cutoff_nb}\n
            Rigid Water:    {self.rigidWater}\n
            Constraints:    {self.constraints}
            """

            self.logger.info(message)

        return system


class SimulationBuilder:
    def __init__(
        self,
        simulation_name: str,
        forcefield: mm.app.forcefield.ForceField,
        integrator: mm.openmm.Integrator,
        system: mm.System,
        modeller: Modeller,
        platform_name: str = "CUDA",
        continue_simulation: bool = False,
        state_xml_path: Union[str, None] = None,
        set_to_temperature: bool = False,
        platform_properties: Union[None, Dict[str, str]] = None,
        do_rest2: bool = False,
        rank: Union[int, None] = None,
        rest2_state_paths: Union[List[str], None] = None,
        alchemical_FE: bool = False,
        scaling_lj_qm_mm: Union[float, None] = None,
        scaling_factor_alchemical_coulomb: Union[float, None] = None,
        logger: Union[None, Logger] = None,
    ) -> None:
        """

        Args:
            simulation_name (str): The name of the simulation.
            forcefield (mm.app.forcefield.ForceField): ForceField object that contains all necessary force fiels for a given system.
            integrator (mm.openmm.Integrator): OpenMM integrator that will be used for simulation
            system (mm.System): OpenMM System
            modeller (Modeller): OpenMM Modeller
            platform_name (str, optional): Platform that will be used for OpenMM. Defaults to "CUDA".
            continue_simulation (bool, optional): Flag to restart the simulation. Defaults to False.
            state_xml_path (Union[str, None], optional): If restarting simulation, set equal to the XML file path of the state from which to restart. Defaults to None.
            set_to_temperature (bool, optional): Assign randomly velocities for all atoms to match specific temperature. Defaults to False.
            platform_properties (Union[None, Dict[str, str]], optional): OpenMM platform property. Use only with multi-GPU REST2 runs. Defaults to None.
            do_rest2 (bool, optional): Flag to do REST2 simulation. Defaults to False.
            rank (Union[int, None], optional): Process rank. Set only if doing REST2. Defaults to None.
            rest2_state_paths (Union[List[str], None], optional): If restart is needed for REST2 run, set equal to the list of XML file paths
            of the states from which to restart. Indexing in the list corresponds to the indexing of the REST2 replicas. Defaults to None.
            alchemical_FE (bool, optional): Flag to do alchemical decoupling of the ML and MM zones. Cannot be done with REST2. Defaults to False.
            scaling_lj_qm_mm (Union[float, None], optional): If alchemical decoupling of the ML and MM zones is done,
            set equal to the lambda factor for LJ decoupling. Defaults to None.
            scaling_factor_alchemical_coulomb (Union[float, None], optional): If alchemical decoupling of the ML and MM zones is done,
            set equal to the lambda factor for electrostatics decoupling. Defaults to None.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.simulation_name = simulation_name
        self.integrator = integrator
        self.forcefield = forcefield
        self.modeller = modeller
        self.platform_name = platform_name
        self.state_xml_path = state_xml_path
        self.continue_simulation = continue_simulation
        self.system = system
        self.set_to_temperature = set_to_temperature
        self.platform_properties = platform_properties
        self.do_rest2 = do_rest2
        self.rank = rank
        self.rest2_state_paths = rest2_state_paths
        self.alchemical_FE = alchemical_FE
        self.scaling_lj_qm_mm = scaling_lj_qm_mm
        self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
        self.logger = logger

        if self.logger:
            self.logger.info(
                f"The simulation object with name {self.simulation_name} is created!"
            )

    def build_simulation(self) -> None:
        """
        Function to create the OpenMM Simulation, set the necessary parameters and if needed load restart state file(s).

        """
        platform = mm.openmm.Platform.getPlatformByName(self.platform_name)

        self.simulation = Simulation(
            topology=self.modeller.topology,
            system=self.system,
            integrator=self.integrator,
            platform=platform,
            platformProperties=self.platform_properties,
        )

        # Continue unbiased MD
        if (
            self.continue_simulation
            and (not self.do_rest2)
            and (not self.alchemical_FE)
        ):
            self.simulation.loadState(self.state_xml_path)

            if self.logger:
                message = f"""The simulation has been continued from the state xml file: {self.state_xml_path}
                """

                self.logger.info(message)

        # Continue REST2 run
        elif self.continue_simulation and self.do_rest2 and (not self.alchemical_FE):
            self.simulation.loadState(self.rest2_state_paths[self.rank])

            if self.logger:
                message = f"""The REST2 simulation has been continued from the state xml files: {self.rest2_state_paths}
                """

                self.logger.info(message)

        # Continue the alchemical decoupling of ML and MM zones
        elif self.continue_simulation and self.alchemical_FE and (not self.do_rest2):
            self.simulation.loadState(self.state_xml_path)

            self.simulation.context.setParameter(
                "scaling_lj_qm_mm", float(self.scaling_lj_qm_mm)
            )

            self.simulation.context.setParameter(
                "scaling_factor_alchemical_coulomb",
                float(self.scaling_factor_alchemical_coulomb),
            )

            if self.logger:
                message = f"""The alchemical FE calulation has been continued from: {self.state_xml_path}
                The scaling_lj_qm_mm is set to: {self.scaling_lj_qm_mm}
                The scaling_factor_alchemical_coulomb is set to: {self.scaling_factor_alchemical_coulomb}
                """

                self.logger.info(message)

        # If not continue, read the initial positions from PDB file
        if not self.continue_simulation:
            self.simulation.context.setPositions(self.modeller.positions)

        # Set the atomic velocities to given temperature
        if self.set_to_temperature:
            self.simulation.context.setVelocitiesToTemperature(
                self.integrator.getTemperature()
            )

            if self.logger:
                message = f"""The velocities has been set to: {self.integrator.getTemperature()}
                """

                self.logger.info(message)

        return self.simulation


class SimulationRunner:
    def __init__(
        self,
        simulation: Simulation,
        steps: int,
        run_type: str,
        save_final_state: bool = False,
        state_xml_path: Union[str, None] = None,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            simulation (Simulation): Valid OpenMM Simulation object
            steps (int): Number of steps to run in the simulation
            run_type (str): Descriptive name for simulation type. Will appear only in logs.
            save_final_state (bool, optional): Flag to save the final state to the XML file.
            Useful if later restart is required.  Defaults to False.
            state_xml_path (Union[str, None], optional): If you want to save the final state, provide
            the path to the XML file, to which the state will be saved. Defaults to None.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.simulation = simulation
        self.steps = steps
        self.run_type = run_type
        self.save_final_state = save_final_state
        self.state_xml_path = state_xml_path
        self.logger = logger

    def run_simulation(self):
        """Function to run the OpenMM simulation."""

        if self.logger:
            message = f"""Starting the {self.run_type} run.
            The simulation length is {self.simulation.integrator.getStepSize() * self.steps}
            """

            self.logger.info(message)

        start_time = time.time()

        self.simulation.step(self.steps)

        end_time = time.time()

        elapsed_time = end_time - start_time

        if self.logger:
            message = f"""The {self.run_type} run is successfull!
            The total time needed for the run:  {elapsed_time} s"""

            self.logger.info(message)

        if self.save_final_state:
            assert self.state_xml_path is not None, (
                "Provide valid state_xml_path to save the state."
            )

            self.simulation.saveState(self.state_xml_path)

            if self.logger:
                message = f"""The final state has been saved to XML file {self.state_xml_path}"""

                self.logger.info(message)


class BarostatAdder(ForcesModifier):
    def __init__(
        self,
        system: mm.System,
        barostat_type: Literal["MCB"],
        barostat_parameters: Dict[str, float],
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            system (mm.System): OpenMM System to which OpenMM Barostat is added.
            barostat_type (Literal[&quot;MCB&quot;]): Abbreviation of OpenMM Barostat.
              Currently only MonteCarloBarostat is supported.
            barostat_parameters (Dict[str, float]): Parameters of the Barostat.
            "pressure" in [bar], "temperature" in [K], "frequency" in [steps]
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.system = system

        self.barostat_type = barostat_type

        self.barostat_parameters = barostat_parameters

        self.logger = logger

    def modify_forces(self) -> mm.System:

        barostat = None

        match self.barostat_type:
            case "MCB":
                pressure = self.barostat_parameters["pressure"] * u.bar
                temp = self.barostat_parameters["temperature"] * u.kelvin
                frequency = self.barostat_parameters["frequency"]

                barostat = mm.openmm.MonteCarloBarostat(pressure, temp, frequency)

        self.system.addForce(barostat)

        if self.logger:
            message = f"""Barostat used: {type(barostat)}\n
            Barostat pressure:  {barostat.getDefaultPressure()}\n
            Barostat temperature:   {barostat.getDefaultTemperature()}
            """
            self.logger.info(message)

        return self.system


class ReporterAdder:
    def __init__(
        self,
        simulation: Simulation,
        readout_frequency: int,
        out_folder_path: str,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            simulation (Simulation): OpenMM Simulation to which reporter(s) will be added.
            readout_frequency (int): The frequency with which the data will be reported in [steps].
            out_folder_path (str): Path to the folder where the reporter files will be written.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.simulation = simulation
        self.readout_frequency = readout_frequency
        self.out_folder_path = out_folder_path
        self.logger = logger

    def write_initial_topology(self, file_name: str) -> None:
        """Function to save the initial topology to the PDB file.
        Be aware that connectivity might change compared to initially read PDB file!

        Args:
            file_name (str): Name of the PDB file (with file extension) to which the initial topology in PDB format will be saved.
        """

        init_topology_path = os.path.join(self.out_folder_path, file_name)

        positions = self.simulation.context.getState(getPositions=True).getPositions()

        PDBFile.writeModel(
            self.simulation.topology, positions, open(init_topology_path, "w")
        )

        PDBFile.writeFooter(self.simulation.topology, open(init_topology_path, "a"))

        if self.logger:
            message = f"""The initial topology is written to: {init_topology_path}"""

            self.logger.info(message)

    def add_dcd_reporter(self, file_name: str, append: bool = False) -> Simulation:
        """Function to add DCD reporter to Simulation

        Args:
            file_name (str): Name of the DCD file (with file extension) to which the coordinates trajectory will be saved.
            append (bool, optional): Set to true if the DCD file exists and you want to append new data to it. Defaults to False.

        Returns:
            Simulation: OpenMM Simulation object with added DCD reporter
        """

        self.simulation.reporters.append(
            mm.app.DCDReporter(
                os.path.join(self.out_folder_path, file_name),
                self.readout_frequency,
                append=append,
            )
        )

        if self.logger:
            message = f"""The DCD reporter is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The DCD file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            """

            self.logger.info(message)

        return self.simulation

    def add_csv_reporter(
        self, file_name: str, parameters: Dict[str, Any], append: bool = False
    ) -> Simulation:
        """Function to add CSV reporter to Simulation

        Args:
            file_name (str): Name of the CSV file (with file extension) to which the parameters trajectory will be saved.
            parameters (Dict[str, Any]): Parameters to configure the CSV reporter. Correspond to OpenMM StateDataReporter arguments.
            append (bool, optional): Set to true if the CSV file exists and you want to append new data to it. Defaults to False.

        Returns:
            Simulation: OpenMM Simulation object with added CSV reporter
        """

        parameters["append"] = append

        self.simulation.reporters.append(
            mm.app.StateDataReporter(
                os.path.join(self.out_folder_path, file_name),
                self.readout_frequency,
                **parameters,
            )
        )

        if self.logger:
            message = f"""The State Data Reporter is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The CSV file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            Properties to be written out: {[key for key, value in parameters.items() if value]}\n
            """

            self.logger.info(message)

        return self.simulation

    def add_hdf5_reporter(
        self,
        file_name: str,
        topology: mm.app.topology.Topology,
        parameters: Dict[str, Any],
        residue_names_to_output: Iterable[str],
    ) -> Simulation:
        """Function to add HDF5 reporter to Simulation

        Args:
            file_name (str): Name of the HDF5 file (with file extension) to which the coordinates and parameters trajectory will be saved.
            topology (mm.app.topology.Topology): Topology of the system
            parameters (Dict[str, Any]): Parameters to configure the HDF5 reporter.
            Correspond to MDTraj HDF5Reporter arguments.
            residue_names_to_output (Iterable[str]): Collection of residue names which positions should be tracked.
            Useful if only subset of residues should be saved (for example, stripping the solvent).
            If all residues should be reported - all residue names must be present in the collection.

        Returns:
            Simulation: OpenMM Simulation object with added HDF5 reporter
        """

        atom_dict = get_atom_indices(residue_names_to_output, topology)

        atomSubset = None

        for key in atom_dict.keys():
            if atomSubset is None:
                atomSubset = atom_dict[key]

            else:
                atomSubset.extend(atom_dict[key])

        atomSubset.sort()

        self.simulation.reporters.append(
            md.reporters.HDF5Reporter(
                file=os.path.join(self.out_folder_path, file_name),
                reportInterval=self.readout_frequency,
                atomSubset=atomSubset,
                **parameters,
            )
        )

        if self.logger:
            message = f"""The HDF5 reporter from MDTraj is added to the simulation.\n
            The readout_frequency is: every {self.readout_frequency} steps.\n
            The .h5 file is written to: {os.path.join(self.out_folder_path, file_name)}\n
            The following residue names are written out: {residue_names_to_output}\n
            """

            self.logger.info(message)

        return self.simulation


class ForcefieldBuilder:
    def __init__(self, forcefield: ForceField, logger: Union[None, Logger] = None):
        """Constructor

        Args:
            forcefield (ForceField): Valid OpenMM ForceField object
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.forcefield = forcefield
        self.logger = logger

    def parametrize_molecules_smirnoff(
        self, molecules: List[Molecule], cache_path: str, ff_name: str
    ) -> None:
        """Function to parametrize the user-defined molecules with OpenFF (SMIRNOFF)

        Args:
            molecules (List[Molecule]): List of OpenFF Molecule objects
            cache_path (str): Path to the JSON file (with file extension), where the generated parameters will be saved.
            ff_name (str): Force field name, corresponds to names used by openmmforcefields.
        """

        smirnoff = SMIRNOFFTemplateGenerator(
            molecules=molecules, cache=cache_path, forcefield=ff_name
        )

        self.forcefield.registerTemplateGenerator(smirnoff.generator)

        if self.logger:
            message = f"""The SMIRNOFFTemplateGenerator is registered.
            Molecular SMILES added to the template generator:    {[mol.to_smiles() for mol in molecules]}\n
            The forcefield used for template generator:     {ff_name}\n
            The forcefdield TemplateGenerator is cached to: {cache_path}\n
            """

            self.logger.info(message)

    def build_forcefield(self) -> ForceField:
        """Get the updated OpenMM force field with parameters for small molecules generated by "parametrize_molecules_smirnoff"."""

        return self.forcefield
