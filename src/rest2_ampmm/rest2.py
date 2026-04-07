""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

import os
import sys
import time
from logging import Logger
from typing import Any, Dict, Iterable, List, Tuple, Union

import mdtraj as md
import numpy as np
import openmm as mm
from mpi4py import MPI
from openmm import unit as u
from openmm.app import PDBFile, Simulation

from rest2_ampmm.helper_functions import get_atom_indices
from rest2_ampmm.utils import ForcesModifier


class ParallelReplicaExchangeRunner:
    def __init__(
        self,
        comm: MPI.Intracomm,
        rank: int,
        parameters: Dict[str, Any],
        system: mm.System,
        simulation: Simulation,
        out_folder_path: str,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            comm (MPI.Intracomm): Check the mpi4py documentation
            rank (int): Process rank
            parameters (Dict[str, Any]): All parameters of AMP simulation
            system (mm.System): Valid OpenMM System
            simulation (Simulation): Valid OpenMM Simulation
            out_folder_path (str): Path to the folder where the output of REST2 run will be saved
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.H_current = rank

        self.comm = comm

        self.rank = rank

        self.parameters = parameters

        self.system = system

        self.simulation = simulation

        self.state_parameters_definition = parameters["state_parameters_definition"]

        self.param_names = parameters["rest2_parameter_names"]

        self.numrep = parameters["numrep"]

        # index of the array means the rank/coordinates/process and value is the hamiltonian index
        self.trajectory_positions = np.array([i for i in range(self.numrep)])

        self.out_folder_path = out_folder_path

        self.logger = logger

        if self.comm.Get_size() < self.numrep:
            raise Exception(
                f"not enough MPI processes ({self.comm.Get_size()} for {self.numrep} replicas)"
            )
        if self.comm.Get_size() > self.numrep:
            # Idle superfluous processes
            if self.rank >= self.numrep:
                self.comm.Barrier()

        self.replica_exchange_changes = []

        self.initialize_exchange_indices()

        self.num_steps_per_iteration = parameters["exchange_frequency"]

        self.num_iterations = int(
            parameters["production_steps"] / self.num_steps_per_iteration
        )

        self.current_iteration = 0

        self.add_reporters(
            residue_names_to_output=self.parameters["residue_names_to_output"]
        )

        if self.rank == 0:
            self.exchange_successfull = False
            self.exchange_stats_path = self.parameters["exchange_stats_path"]

    def initialize_exchange_indices(self) -> None:
        """Helper function to define the pairs of replica indices that are used for exchange for even and odd iterations."""

        even = [
            (i, i + 1)
            for i in range(self.numrep)
            if (i % 2 == 0) and (i != self.numrep - 1)
        ]
        odd = [
            (i, i + 1)
            for i in range(self.numrep)
            if (i % 2 != 0) and (i != self.numrep - 1)
        ]
        excluded_idx_even = [
            i for i in range(self.numrep) if i not in [j for sub in even for j in sub]
        ]
        excluded_idx_odd = [
            i for i in range(self.numrep) if i not in [j for sub in odd for j in sub]
        ]

        self.index_lists = [even, odd]
        self.excluded_indices = [excluded_idx_even, excluded_idx_odd]

        if self.logger:
            message = f"""Even exchange indices are {even}\n
            Odd exchange indices are {odd}.
            """

            self.logger.info(message)

    def setStateParameters(self, index: int) -> None:
        """Helper function to set the REST2 parameters to a given state by index

        Args:
            index (int): State index
        """

        for param_name in self.param_names:
            self.simulation.context.setParameter(
                param_name, self.state_parameters_definition[index][param_name]
            )

    def make_rest2_iteration(self) -> None:
        """Do the MD steps for 1 REST2 iteration"""

        self.simulation.step(self.num_steps_per_iteration)

    def calculate_exchange_probability(
        self, h1x1: float, h1x2: float, h2x1: float, h2x2: float
    ) -> float:
        """Function to calculate the exchange probability for 2 Hamiltonians and 2 sets of coordinates

        Args:
            h1x1 (float): Energy of coordinates set 1 with Hamiltonian 1
            h1x2 (float): Energy of coordinates set 2 with Hamiltonian 1
            h2x1 (float): Energy of coordinates set 1 with Hamiltonian 2
            h2x2 (float): Energy of coordinates set 2 with Hamiltonian 2

        Returns:
            float: Probability of replica exchange
        """

        temp = self.parameters["integrator_parameters"]["temperature"] * u.kelvin

        # calculate beta
        beta = 1 / (u.BOLTZMANN_CONSTANT_kB * temp)

        # calculate probability
        deltaH1 = h1x1 - h1x2
        deltaH2 = h2x1 - h2x2

        p = np.exp((beta * deltaH1 - beta * deltaH2) / u.AVOGADRO_CONSTANT_NA)

        prob = min(1, p)

        return prob

    def _handle_master_exchange(
        self, rank_1: int, rank_2: int, hamiltonian_1: int, hamiltonian_2: int
    ) -> None:
        """Function to perform the replica exchange and update the rank-Hamiltonian map involving the master node (rank 0)

        Args:
            rank_1 (int): Process rank 1 participating in exchange
            rank_2 (int): Process rank 2 participating in exchange
            hamiltonian_1 (int): Hamiltonian index 1 participating in exchange (will be set to rank 2 if exchange accepted, else to rank 1)
            hamiltonian_2 (int): Hamiltonian index 2 participating in exchange (will be set to rank 1 if exchange accepted, else to rank 2)
        """
        if rank_1 == 0:
            h1x1 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()

            self.comm.send(hamiltonian_1, dest=rank_2)

            self.setStateParameters(index=hamiltonian_2)

            h2x1 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()

            h2x2, h1x2 = self.comm.recv(source=rank_2)

        else:
            h2x2 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()

            self.comm.send(hamiltonian_2, dest=rank_1)

            self.setStateParameters(index=hamiltonian_1)

            h1x2 = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()

            h1x1, h2x1 = self.comm.recv(source=rank_1)

        prob = self.calculate_exchange_probability(h1x1, h1x2, h2x1, h2x2)

        self.exchange_successfull = False
        if prob > np.random.rand():
            if rank_1 == 0:
                self.comm.send(hamiltonian_1, dest=rank_2)

                self.setStateParameters(index=hamiltonian_2)
                self.H_current = hamiltonian_2
            else:
                self.comm.send(hamiltonian_2, dest=rank_1)

                self.setStateParameters(index=hamiltonian_1)
                self.H_current = hamiltonian_1

            self.trajectory_positions[rank_1] = hamiltonian_2
            self.trajectory_positions[rank_2] = hamiltonian_1

            self.exchange_successfull = True
            self.exchange_success_per_iteration.append(
                (hamiltonian_1, hamiltonian_2, self.exchange_successfull)
            )

        else:
            if rank_1 == 0:
                self.comm.send(hamiltonian_2, dest=rank_2)

                self.setStateParameters(index=hamiltonian_1)

                self.H_current = hamiltonian_1
            else:
                self.comm.send(hamiltonian_1, dest=rank_1)

                self.setStateParameters(index=hamiltonian_2)

                self.H_current = hamiltonian_2

            self.trajectory_positions[rank_1] = hamiltonian_1

            self.trajectory_positions[rank_2] = hamiltonian_2

            self.exchange_success_per_iteration.append(
                (hamiltonian_1, hamiltonian_2, self.exchange_successfull)
            )

    def perform_replica_exchange(
        self, rank_i: int, rank_j: int, hamiltonian_1: int, hamiltonian_2: int
    ) -> None:
        """Function to perform the replica exchange and update the rank-Hamiltonian map

        Args:
            rank_i (int): Process rank 1 participating in exchange
            rank_j (int): Process rank 2 participating in exchange
            hamiltonian_1 (int): Hamiltonian index 1 participating in exchange (will be set to rank 1)
            hamiltonian_2 (int): Hamiltonian index 2 participating in exchange (will be set to rank 2)
        """

        self.comm.send(hamiltonian_1, dest=rank_i)

        self.comm.send(hamiltonian_2, dest=rank_j)

        self.trajectory_positions[rank_i] = hamiltonian_1

        self.trajectory_positions[rank_j] = hamiltonian_2

    def perform_replica_exchanges(
        self, indices: List[Tuple[int, int]], excluded_indices: List[int]
    ) -> None:
        """Main function that handles all replica exchanges in REST2. Implements the logic for master and slave processes.

        Args:
            indices (List[Tuple[int, int]]): Pairs of Hamiltonian indices for which the replica exchange should be attempted
            excluded_indices (List[int]): Hamiltonian indices that are excluded from an attempt
              of an exchange due to the absence of the neighboring index
        """

        # rank 0 calculates exchange probabilities and sends new parameter values to other ranks
        if self.rank == 0:
            self.exchange_success_per_iteration = []

            for hamiltonian_indices in indices:
                hamiltonian_1 = hamiltonian_indices[0]

                hamiltonian_2 = hamiltonian_indices[1]

                self.exchange_successfull = False

                # Determine ranks associated with the Hamiltonians
                rank_1 = np.where(self.trajectory_positions == hamiltonian_1)[0][0]

                rank_2 = np.where(self.trajectory_positions == hamiltonian_2)[0][0]

                if rank_1 == 0 or rank_2 == 0:  # Special case involving the master node
                    self._handle_master_exchange(
                        rank_1, rank_2, hamiltonian_1, hamiltonian_2
                    )

                else:  # Normal case
                    # send message to slaves to compute energies for hamiltonian 1 and 2
                    for rank, H_i in zip(
                        [rank_2, rank_1], [hamiltonian_1, hamiltonian_2]
                    ):
                        self.comm.send(H_i, dest=rank)

                    # calculate the energies
                    h1x1, h2x1 = self.comm.recv(source=rank_1)

                    h2x2, h1x2 = self.comm.recv(source=rank_2)

                    prob = self.calculate_exchange_probability(h1x1, h1x2, h2x1, h2x2)

                    if prob > np.random.rand(1):
                        self.perform_replica_exchange(
                            rank_2, rank_1, hamiltonian_1, hamiltonian_2
                        )

                        self.exchange_successfull = True

                        self.exchange_success_per_iteration.append(
                            (hamiltonian_1, hamiltonian_2, self.exchange_successfull)
                        )

                    else:
                        self.perform_replica_exchange(
                            rank_1, rank_2, hamiltonian_1, hamiltonian_2
                        )

                        self.exchange_success_per_iteration.append(
                            (hamiltonian_1, hamiltonian_2, self.exchange_successfull)
                        )

        # ranks != 0 receive their current parameter values, calculate the energies and return them to master
        else:
            if self.H_current not in excluded_indices:
                h_current_x_current = self.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()

                H_new = self.comm.recv(source=0)

                self.setStateParameters(index=H_new)

                h_new_x_current = self.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()

                self.comm.send([h_current_x_current, h_new_x_current], dest=0)

                # Set the Hamiltonian for propagation
                H_propagation = self.comm.recv(source=0)

                self.setStateParameters(index=H_propagation)

                self.H_current = H_propagation

            else:
                self.H_current = self.H_current

    def save_state(self) -> None:
        """Function to save the state of the simulation to the checkpoint XML file, synchronized across all replicas"""

        # Wait for all other replicas to finish the exchange
        self.comm.Barrier()

        self.simulation.saveState(
            os.path.join(
                self.out_folder_path, f"checkpoint_replica_{self.H_current:04d}.xml"
            )
        )

    def save_initial_state(self) -> None:
        """Function to save the current simulation state to the checkpoint XML file"""

        self.simulation.saveState(
            os.path.join(
                self.out_folder_path, f"checkpoint_replica_{self.H_current:04d}.xml"
            )
        )

    def write_initial_topology(self, file_name: str) -> None:
        """Function to save system topology to the PDB file

        Args:
            file_name (str): PDB file name (with file extension) where the system topology will be saved
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

    def add_reporters(
        self, append: bool = False, residue_names_to_output: Iterable[str] = None
    ) -> None:
        """Function to add the HDF5 reporter to the simulation

        Args:
            append (bool, optional): Set to true if the HDF5 already exists and you want to append to it. Defaults to False.
            residue_names_to_output (Iterable[str], optional): Collection of residue names which positions should be tracked.
            If None all positions are tracked. Defaults to None.

        Raises:
            FileNotFoundError: If HDF5 file is does not exist and append=True
        """

        if not append and self.rank == 0:
            self.write_initial_topology(file_name="init_topology.pdb")

        h5_path = os.path.join(self.out_folder_path, f"replica_{self.H_current:02d}.h5")

        if residue_names_to_output is None:
            atomSubset = None

        else:
            atom_dict = get_atom_indices(
                residue_names_to_output, self.simulation.topology
            )
            atomSubset = []
            for key in atom_dict.keys():
                atomSubset.extend(atom_dict[key])
            atomSubset.sort()

        # Decide whether to open file for appending or create new
        if append:
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"The file {h5_path} does not exist, so it cannot be appended!"
                )
            file = md.formats.HDF5TrajectoryFile(h5_path, mode="a")
        else:
            file = h5_path  # create new file

        self.simulation.reporters.append(
            md.reporters.HDF5Reporter(
                file=file,
                reportInterval=self.num_steps_per_iteration,
                atomSubset=atomSubset,
                coordinates=True,
                time=True,
                cell=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                velocities=False,
            )
        )

    def update_reporters(self) -> None:
        """Function to update the reporters after replica exchange to keep the HDF5 files in
        order with the Hamiltonian (HDF5 file has trajectory of the same Hamiltonian throughout the simulation)"""

        self.simulation.reporters.clear()

        self.comm.Barrier()

        self.add_reporters(
            append=True,
            residue_names_to_output=self.parameters["residue_names_to_output"],
        )

    def save_rest2_stats(self) -> None:
        """Helper function to save the replica exhcnage statistics to the file"""

        with open(self.exchange_stats_path, "a") as file:
            res = " ".join([str(i) for i in self.exchange_success_per_iteration])
            string_to_write = f"Iteration   {self.current_iteration}:    {res}   \n\n"
            file.write(string_to_write)

    def run_simulation(self) -> None:
        """The main function of the parallel REST2 simulation."""

        if self.logger:
            message = f"""Starting the REST2 run.
            The total number of steps is: {self.parameters["production_steps"]}\n
            The exchange frequency is every {self.num_steps_per_iteration} steps\n
            The number total number of iterations = total_steps/exchange_freq = {self.num_iterations}\n
            The number of replicas is {self.parameters["numrep"]}\n
            The scaling factor names are: {self.param_names}\n\n

            """

            self.logger.info(message)

            for i, val in enumerate(self.state_parameters_definition):
                lines = [f"Replica {i}:"]
                for key, value in val.items():
                    lines.append(f"  {key} {value:.3f}")

                message = "\n".join(lines)
                self.logger.info(message)

            if self.parameters["rest2_save_checkpoints"]:
                message = "The checkpoints for every replica will be saved separately and overwritten every iteration.\n"

                self.logger.info(message)

        start_time = time.time()

        failed = False

        for i in range(self.num_iterations):
            try:
                self.make_rest2_iteration()

            except Exception as e:
                message = (
                    f"Error: Replica {self.rank} crashed! Aborting the REST2 run! {e}"
                )

                print(message, flush=True)

                failed = True

            if self.comm.allreduce(failed, op=MPI.LOR):
                sys.stdout.flush()

                self.comm.Barrier()

                self.comm.Abort()

            indices = self.index_lists[i % 2]

            excluded_indices = self.excluded_indices[i % 2]

            self.save_initial_state()  # R

            self.perform_replica_exchanges(indices, excluded_indices)

            self.save_state()

            self.update_reporters()

            if self.rank == 0:
                self.save_rest2_stats()

                self.current_iteration = self.current_iteration + 1

        end_time = time.time()

        elapsed_time = end_time - start_time

        if self.logger:
            message = f"""The REST2 run is successfull!
            The total time needed for the run:  {elapsed_time} s"""

            self.logger.info(message)


class HarmonicRestraintsForREST2Adder(ForcesModifier):
    def __init__(
        self,
        system: mm.System,
        topology: mm.app.topology.Topology,
        residue_names: Iterable[str],
        scale_factor_bond_rest2: float,
        scale_factor_angle_rest2: float,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            system (mm.System): Valid OpenMM System
            topology (mm.app.topology.Topology): Valid OpenMM Topology
            residue_names (Iterable[str]): Collection of residue names that define the ML zone and need special harmonic restraints in REST2
            scale_factor_bond_rest2 (float): Scaling factor of the force constant for harmonic potential of bonds
            scale_factor_angle_rest2 (float): Scaling factor of the force constant for harmonic potential of angles
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.system = system
        self.topology = topology
        self.residue_names = residue_names
        self.scale_factor_bond_rest2 = scale_factor_bond_rest2
        self.scale_factor_angle_rest2 = scale_factor_angle_rest2
        self.logger = logger

    def modify_forces(self) -> mm.System:
        """Function to add the harmonic potentials to bonds and angles in the given residues for REST2 run.

        Returns:
            mm.System: Updated OpenMM System with the harmonic potentials for bonds and angles in the given residues for REST2 run
        """

        self._get_forces()

        self._get_atomic_indices_involved()

        self.add_harmonic_restraints_for_all_bonds()

        self.add_harmonic_restraints_for_all_angles()

        return self.system

    def _get_forces(self) -> None:
        """Helper function to get the existing harmonic bond and angle forces"""

        for idf, force in enumerate(self.system.getForces()):
            if isinstance(force, mm.openmm.HarmonicBondForce):
                self.bond_force = force
                self.id_bond_force = idf

            elif isinstance(force, mm.openmm.HarmonicAngleForce):
                self.angle_force = force
                self.id_angle_force = idf

    def _get_atomic_indices_involved(self) -> None:
        """Helper function to get all atomic indices of the residues that should me modified"""
        dict_of_atomic_indices = get_atom_indices(self.residue_names, self.topology)

        all_atomic_indices = set()

        for value in dict_of_atomic_indices.values():
            all_atomic_indices.update(value)

        self.all_atomic_indices = all_atomic_indices

    def add_harmonic_restraints_for_all_bonds(
        self,
        expression: str = "0.5*k_restraint_bond*scale_factor_bond_rest2*(r-eq_distance)^2",
    ) -> None:
        """Function to add harmonic forces for all bonds in requested residues. Forces are added for all pairs of atoms
         that form a bond in the initial topology in requested residue names.

        Args:
            expression (str, optional): Expression of the harmonic force to be used in OpenMM. Defaults to "0.5*k_restraint_bond*scale_factor_bond_rest2*(r-eq_distance)^2".
        """
        restraining_bond_force = mm.openmm.CustomBondForce(expression)
        restraining_bond_force.addPerBondParameter("k_restraint_bond")
        restraining_bond_force.addPerBondParameter("eq_distance")
        restraining_bond_force.addGlobalParameter(
            "scale_factor_bond_rest2", self.scale_factor_bond_rest2
        )

        for bond_id in range(self.bond_force.getNumBonds()):
            bond_id_1, bond_id_2, eq_distance, k_constant = (
                self.bond_force.getBondParameters(bond_id)
            )

            if any(
                (
                    bond_id_1 not in self.all_atomic_indices,
                    bond_id_2 not in self.all_atomic_indices,
                )
            ):
                continue

            else:
                restraining_bond_force.addBond(
                    bond_id_1, bond_id_2, [k_constant / 1.2, eq_distance]
                )

        self.system.addForce(restraining_bond_force)

        if self.logger:
            message = """Harmonic bond forces in the QM zone have been doubled using the CustomBondForce!"""

            self.logger.info(message)

    def add_harmonic_restraints_for_all_angles(
        self,
        expression: str = "0.5*k_restraint_angle*scale_factor_angle_rest2*(theta-eq_angle)^2",
    ):
        """Function to add harmonic forces for all angles in requested residues. Forces are added for all triplets of atoms
         that form an angle in the initial topology in requested residue names.

        Args:
            expression (str, optional): Expression of the harmonic force to be used in OpenMM. Defaults to "0.5*k_restraint_bond*scale_factor_bond_rest2*(r-eq_distance)^2".
        """
        restraining_angle_force = mm.openmm.CustomAngleForce(expression)
        restraining_angle_force.addPerAngleParameter("k_restraint_angle")
        restraining_angle_force.addPerAngleParameter("eq_angle")
        restraining_angle_force.addGlobalParameter(
            "scale_factor_angle_rest2", self.scale_factor_angle_rest2
        )

        for angle_id in range(self.angle_force.getNumAngles()):
            angle_id_1, angle_id_2, angle_id_3, eq_angle, k_constant = (
                self.angle_force.getAngleParameters(angle_id)
            )

            if any(
                (
                    angle_id_1 not in self.all_atomic_indices,
                    angle_id_2 not in self.all_atomic_indices,
                    angle_id_3 not in self.all_atomic_indices,
                )
            ):
                continue

            else:
                restraining_angle_force.addAngle(
                    angle_id_1, angle_id_2, angle_id_3, [k_constant / 1.2, eq_angle]
                )

        self.system.addForce(restraining_angle_force)

        if self.logger:
            message = """Harmonic angle forces in the QM zone have been doubled using the CustomAngleForce!"""
            self.logger.info(message)
