""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

from logging import Logger
from typing import Iterable, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import openmm as mm
import torch
from openmm import unit as u
from openmmtorch import TorchForce
from torch.jit import ScriptModule

from rest2_ampmm.bioff.amp.AMP import AMP
from rest2_ampmm.bioff.datastructures.Graphs import Graph
from rest2_ampmm.bioff.utilities.Helpers import load_parameters
from rest2_ampmm.bioff.utilities.Utilities import build_Rx2
from rest2_ampmm.helper_functions import get_atom_indices
from rest2_ampmm.utils import ForcesModifier


class AmpForcesModifier(ForcesModifier):
    def __init__(
        self,
        system: mm.System,
        topology: mm.app.topology.Topology,
        qm_zone: npt.NDArray[np.int_],
        mm_zone: npt.NDArray[np.int_],
        eps_rf: float,
        tip4p: bool,
        cutoff_nb: mm.unit.quantity.Quantity,
        scaling_lj_qm_mm: float = 1.0,
        softcore_lj_qm_mm: bool = False,
        logger: Union[None, Logger] = None,
    ) -> None:
        """Constructor

        Args:
            system (mm.System): Valid OpenMM System, the forces of which will be modified to be compatible with AMP usage
            topology (mm.app.topology.Topology): Valid OpenMM topology
            qm_zone (npt.NDArray[np.int_]): QM zone atomic indices (0-based)
            mm_zone (npt.NDArray[np.int_]): MM zone atomic indices (0-based)
            eps_rf (float): Dielectric permittivity of the medium, used for Reaction Field, unitless
            tip4p (bool): Set True if the MM zone contains TIP4P molecules
            cutoff_nb (mm.unit.quantity.Quantity): The nonbonded cutoff value in [nm]
            scaling_lj_qm_mm (float, optional): Lambda for the ML-MM Lennard-Jones interactions
            scaling in alchemical decoupling. Defaults to 1.0.
            softcore_lj_qm_mm (bool, optional): Set True if the usual Lennard-Jones potential for ML-MM interaction
            should be substituted with softcore version in alchemical decoupling. Defaults to False.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.topology = topology

        self.system = system

        self.logger = logger

        self.qm_zone = qm_zone

        self.mm_zone = mm_zone

        self.tip4p = tip4p

        self.eps_rf = eps_rf

        self.cutoff_nb = cutoff_nb

        self.scaling_lj_qm_mm = scaling_lj_qm_mm

        self.softcore_lj_qm_mm = softcore_lj_qm_mm

        self.softcore_alpha = 0.5

        self._get_forces()

    def _get_forces(self) -> None:
        """Helper function to collect all existing standard forces in the OpenMM System"""

        for idf, force in enumerate(self.system.getForces()):
            if isinstance(force, mm.openmm.HarmonicBondForce):
                self.bond_force = force
                self.id_bond_force = idf
            elif isinstance(force, mm.openmm.NonbondedForce):
                self.nb_force = force
                self.id_nb_force = idf
            elif isinstance(force, mm.openmm.PeriodicTorsionForce):
                self.torsion_force = force
                self.id_torsion_force = idf
            elif isinstance(force, mm.openmm.HarmonicAngleForce):
                self.angle_force = force
                self.id_angle_force = idf

    def _build_custom_nonbonded_mmmm(self) -> None:
        """Function to define the custom Reaction Field and Lennard-Jones potentials for MM-MM interactions"""

        krf = ((self.eps_rf - 1) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb**3)
        ONE_4PI_EPS0 = 138.935456  # * u.kilojoules_per_mole*u.nanometer/(u.elementary_charge_base_unit*u.elementary_charge_base_unit)
        mrf = 4
        nrf = 6
        arfm = (3 * self.cutoff_nb ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * (
            (2 * self.eps_rf + nrf - 1) / (1 + 2 * self.eps_rf)
        )
        arfn = (3 * self.cutoff_nb ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * (
            (2 * self.eps_rf + mrf - 1) / (1 + 2 * self.eps_rf)
        )
        crf = (
            ((3 * self.eps_rf) / (1 + 2 * self.eps_rf)) * (1 / self.cutoff_nb)
            + arfm * self.cutoff_nb**mrf
            + arfn * self.cutoff_nb**nrf
        )

        # Custom reaction field expression
        crf_exp = "ONE_4PI_EPS0*chargeprod*(1/r + krf*r2 + arfm*r4 + arfn*r6 - crf);"
        crf_exp += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        crf_exp += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        crf_exp += "r6 = r2*r4;"
        crf_exp += "r4 = r2*r2;"
        crf_exp += "r2 = r*r;"
        crf_exp += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        crf_exp += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))
        crf_exp += "chargeprod = charge1*charge2;"
        crf_exp += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)

        # Lennard-Jones expression
        lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
        lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        lj_exp += "sigma_over_r = sigma/r;"
        lj_exp += "epsilon = sqrt(epsilon1*epsilon2);"
        lj_exp += "sigma = 0.5*(sigma1+sigma2);"

        # 1-4 interactions for custom reaction field and LJ potential
        lj_crf_one_four = "(4*epsilon*(sigma_over_r12 - sigma_over_r6) + ONE_4PI_EPS0*chargeprod*(1/r) +ONE_4PI_EPS0*chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 - crf));"
        lj_crf_one_four += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        lj_crf_one_four += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        lj_crf_one_four += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        lj_crf_one_four += "r6 = r2*r4;"
        lj_crf_one_four += "r4 = r2*r2;"
        lj_crf_one_four += "r2 = r*r;"
        lj_crf_one_four += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
        lj_crf_one_four += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
        lj_crf_one_four += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
        lj_crf_one_four += "sigma_over_r = sigma/r;"
        lj_crf_one_four += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        lj_crf_one_four += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))

        # Excluded term for CRF
        crf_excluded = "(ONE_4PI_EPS0*chargeprod_*(krf*r2 + arfm*r4 + arfn*r6 -crf));"
        crf_excluded += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        crf_excluded += "krf = {:f};".format(krf.value_in_unit(u.nanometer**-3))
        crf_excluded += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))
        crf_excluded += "r6 = r2*r4;"
        crf_excluded += "r4 = r2*r2;"
        crf_excluded += "r2 = r*r;"
        crf_excluded += "arfm = {:f};".format(arfm.value_in_unit(u.nanometer**-5))
        crf_excluded += "arfn = {:f};".format(arfn.value_in_unit(u.nanometer**-7))

        # Self term for CRF
        crf_self_term = "(0.5 * ONE_4PI_EPS0* chargeprod_ * (-crf));"
        crf_self_term += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
        crf_self_term += "crf = {:f};".format(crf.value_in_unit(u.nanometer**-1))

        force_crf = mm.CustomNonbondedForce(crf_exp)
        force_crf.addPerParticleParameter("charge")
        force_crf.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_crf.setCutoffDistance(self.cutoff_nb)
        force_crf.setName("custom_RF_mm-mm")

        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        if self.tip4p:
            force_lj.setUseLongRangeCorrection(True)
        else:
            force_lj.setUseLongRangeCorrection(False)
        force_lj.setName("lj_mm-mm")

        force_lj_crf_one_four = mm.CustomBondForce(lj_crf_one_four)
        force_lj_crf_one_four.addPerBondParameter("chargeprod")
        force_lj_crf_one_four.addPerBondParameter("sigma")
        force_lj_crf_one_four.addPerBondParameter("epsilon")
        force_lj_crf_one_four.addPerBondParameter("chargeprod_")
        force_lj_crf_one_four.setName("lj_crf_one_four_mm-mm")

        force_crf_excluded = mm.CustomBondForce(crf_excluded)
        force_crf_excluded.addPerBondParameter("chargeprod_")
        force_crf_excluded.setName("custom_RF_excl_mm-mm")

        force_crf_self_term = mm.CustomBondForce(crf_self_term)
        force_crf_self_term.addPerBondParameter("chargeprod_")
        force_crf_self_term.setName("custom_RF_self_int_mm-mm")

        for index in range(self.nb_force.getNumParticles()):
            charge, sigma, epsilon = self.nb_force.getParticleParameters(index)

            force_crf.addParticle([charge])
            force_lj.addParticle([sigma, epsilon])

        # Retain exceptions for mm-mm interactions
        for index in range(self.nb_force.getNumExceptions()):
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(
                index
            )

            if j in self.mm_zone and k in self.mm_zone:
                force_crf.addExclusion(j, k)
                force_lj.addExclusion(j, k)

        # Treat MM-MM exceptions as a separate force
        for index in range(self.nb_force.getNumExceptions()):
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(
                index
            )

            if (
                j in self.mm_zone
                and k in self.mm_zone
                and (chargeprod._value != 0 or epsilon._value != 0)
            ):
                ch1, _, _ = self.nb_force.getParticleParameters(j)
                ch2, _, _ = self.nb_force.getParticleParameters(k)
                force_lj_crf_one_four.addBond(
                    j, k, [chargeprod, sigma, epsilon, ch1 * ch2]
                )

        # treat MM-MM exclusions as a separate force
        for index in range(self.nb_force.getNumExceptions()):
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(
                index
            )

            if j in self.mm_zone and k in self.mm_zone and (chargeprod._value == 0):
                ch1, _, _ = self.nb_force.getParticleParameters(j)
                ch2, _, _ = self.nb_force.getParticleParameters(k)
                force_crf_excluded.addBond(j, k, [ch1 * ch2])

        for index in range(self.nb_force.getNumParticles()):
            if index not in self.mm_zone:
                continue

            charge, sigma, epsilon = self.nb_force.getParticleParameters(index)

            force_crf_self_term.addBond(index, index, [charge * charge])

        force_lj.addInteractionGroup(set(self.mm_zone), set(self.mm_zone))
        force_crf.addInteractionGroup(set(self.mm_zone), set(self.mm_zone))

        self.system.addForce(force_lj)
        self.system.addForce(force_crf)
        self.system.addForce(force_lj_crf_one_four)
        self.system.addForce(force_crf_excluded)
        self.system.addForce(force_crf_self_term)

        if self.logger:
            message = """The long-range electrostatics method has been set to reaction field!
           The MM-MM LJ and custom RF interactions have been added to the system!"""

            self.logger.info(message)

    def _build_custom_nonbonded_qmmm(self) -> None:
        """Function to define the Lennard-Jones potential for ML-MM interactions. If needed, uses the softcore implementation."""

        if self.softcore_lj_qm_mm:
            lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
            lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
            lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
            lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
            lj_exp += "sigma_over_r = sigma/reff_vdw;"
            lj_exp += "epsilon = sqrt(epsilon1*epsilon2)*scaling_lj_qm_mm;"
            lj_exp += f"reff_vdw = sigma*({self.softcore_alpha}*(1-scaling_lj_qm_mm) + (r/sigma)^6)^(1/6);"
            lj_exp += "sigma = 0.5*(sigma1+sigma2);"

        else:
            lj_exp = "4*epsilon*(sigma_over_r12 - sigma_over_r6);"
            lj_exp += "sigma_over_r12 = sigma_over_r6 * sigma_over_r6;"
            lj_exp += "sigma_over_r6 = sigma_over_r3 * sigma_over_r3;"
            lj_exp += "sigma_over_r3 = sigma_over_r * sigma_over_r * sigma_over_r;"
            lj_exp += "sigma_over_r = sigma/r;"
            lj_exp += "epsilon = sqrt(epsilon1*epsilon2)*scaling_lj_qm_mm;"
            lj_exp += "sigma = 0.5*(sigma1+sigma2);"

        force_lj = mm.CustomNonbondedForce(lj_exp)
        force_lj.addPerParticleParameter("sigma")
        force_lj.addPerParticleParameter("epsilon")
        force_lj.addGlobalParameter("scaling_lj_qm_mm", self.scaling_lj_qm_mm)
        force_lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force_lj.setCutoffDistance(self.cutoff_nb)
        force_lj.setName("lj_qm-mm")
        if self.cutoff_nb < 1.0 * u.nanometer:
            force_lj.setUseLongRangeCorrection(True)  # for TIP4P-FB
        else:
            force_lj.setUseLongRangeCorrection(False)  # for TIP3P and OPENFF

        for index in range(self.nb_force.getNumParticles()):
            _, sigma, epsilon = self.nb_force.getParticleParameters(index)

            force_lj.addParticle([sigma, epsilon])

        force_lj.addInteractionGroup(set(self.qm_zone), set(self.mm_zone))

        # Retain exceptions for mm-mm interactions
        for index in range(self.nb_force.getNumExceptions()):
            j, k, chargeprod, sigma, epsilon = self.nb_force.getExceptionParameters(
                index
            )

            if j in self.mm_zone and k in self.mm_zone:
                force_lj.addExclusion(j, k)

        self.system.addForce(force_lj)

        if self.logger:
            message = """The QM-MM LJ interactions have been added to the system!"""

            self.logger.info(message)

    def _zero_bond_force_qm_zone(self) -> None:
        """Function to zero-down all harmonic bonded forces inside the ML zone to avoid duplication with AMP."""

        zero_force_constant = 0.0

        for bond_id in range(self.bond_force.getNumBonds()):
            bond_id_1, bond_id_2, eq_distance, k_constant = (
                self.bond_force.getBondParameters(bond_id)
            )

            if any((bond_id_1 not in self.qm_zone, bond_id_2 not in self.qm_zone)):
                continue

            else:
                self.bond_force.setBondParameters(
                    bond_id, bond_id_1, bond_id_2, eq_distance, zero_force_constant
                )

        if self.logger:
            message = """Harmonic bond forces in the QM zone have been zeroed!"""

            self.logger.info(message)

    def _zero_angle_force_qm_zone(self) -> None:
        """Function to zero-down all harmonic angle forces inside the ML zone to avoid duplication with AMP."""

        zero_force_constant = 0.0

        for angle_id in range(self.angle_force.getNumAngles()):
            angle_id_1, angle_id_2, angle_id_3, eq_distance, k_constant = (
                self.angle_force.getAngleParameters(angle_id)
            )

            if any(
                (
                    angle_id_1 not in self.qm_zone,
                    angle_id_2 not in self.qm_zone,
                    angle_id_3 not in self.qm_zone,
                )
            ):
                continue

            else:
                self.angle_force.setAngleParameters(
                    angle_id,
                    angle_id_1,
                    angle_id_2,
                    angle_id_3,
                    eq_distance,
                    zero_force_constant,
                )

        if self.logger:
            message = """Harmonic angle forces in the QM zone have been zeroed!"""
            self.logger.info(message)

    def _zero_torsion_force_qm_zone(self) -> None:
        """Function to zero-down all torsion forces inside the ML zone to avoid duplication with AMP."""

        zero_force_constant = 0.0

        for torsion_id in range(self.torsion_force.getNumTorsions()):
            (
                torsion_id_1,
                torsion_id_2,
                torsion_id_3,
                torsion_id_4,
                periodicity,
                shift,
                k_constant,
            ) = self.torsion_force.getTorsionParameters(torsion_id)

            if any(
                (
                    torsion_id_1 not in self.qm_zone,
                    torsion_id_2 not in self.qm_zone,
                    torsion_id_3 not in self.qm_zone,
                    torsion_id_4 not in self.qm_zone,
                )
            ):
                continue

            else:
                self.torsion_force.setTorsionParameters(
                    torsion_id,
                    torsion_id_1,
                    torsion_id_2,
                    torsion_id_3,
                    torsion_id_4,
                    periodicity,
                    shift,
                    zero_force_constant,
                )

        if self.logger:
            message = """Torsion forces in the QM zone have been zeroed!"""
            self.logger.info(message)

    def _remove_old_nonbonded_force(self) -> None:
        """Function to remove the old (initially defined) Nonbonded Force (for all ML-ML, ML-MM and MM-MM interactions)
        to avoid duplication with AMP."""

        self.system.removeForce(self.id_nb_force)

        if self.logger:
            message = """Old Nonbonded Force has been removed!"""
            self.logger.info(message)

    def modify_forces(self) -> mm.System:
        """Function to get the updated OpenMM System with all initially existing forces modified to be compatible with AMP usage.

        Returns:
            mm.System: Updated System
        """

        self._zero_bond_force_qm_zone()
        self._zero_angle_force_qm_zone()
        self._zero_torsion_force_qm_zone()
        self._build_custom_nonbonded_mmmm()
        self._build_custom_nonbonded_qmmm()
        self._remove_old_nonbonded_force()

        return self.system


class AmpTorchForceAdder(ForcesModifier):
    def __init__(
        self,
        system: mm.System,
        topology: mm.app.topology.Topology,
        qm_zone: npt.NDArray[np.int_],
        charges_mm: npt.NDArray[np.floating],
        mm_zone_charges: npt.NDArray[np.int_],
        params_path: str,
        weights_path: str,
        device_ml: str,
        scaling_factor_node_potential: float,
        scaling_factor_coulomb_qm: float,
        scaling_factor_coulomb_qmmm: float,
        scaling_factor_D4: float,
        scaling_factor_ZBL: float,
        mol_charge: int = 0,
        scaling_charges: float = 1.0,
        scaling_factor_alchemical_coulomb: float = 1.0,
        rank: Union[int, None] = None,
        logger: Union[None, Logger] = None,
    ):
        """Constructor

        Args:
            system (mm.System): valid OpenMM System to which the AMP force will be added
            topology (mm.app.topology.Topology): Valid OpenMM Topology for the System of interest
            qm_zone (npt.NDArray[np.int_]): Indices of the ML zone atoms (0-based)
            charges_mm (npt.NDArray[np.floating]): Charges of the atoms in MM zone
            mm_zone_charges (npt.NDArray[np.int_]): Indices of the MM zone atoms with non-zero charges
            params_path (str): Path to the YAML file with AMP configuration
            weights_path (str): Path to the AMP weights file
            device_ml (str): Device to be used for AMP model, corresponds to PyTorch device
            scaling_factor_node_potential (float): Scaling factor for node potential, used in REST2
            scaling_factor_coulomb_qm (float): Scaling factor for ML-ML electrostatic interaction, used in REST2
            scaling_factor_coulomb_qmmm (float): Scaling factor for ML-MM electrostatic interaction, used in REST2
            scaling_factor_D4 (float): Scaling factor dispersion potential (ML-ML interaction, D4 potential), used in REST2
            scaling_factor_ZBL (float): Scaling factor ZBL potential (ML-ML interaction), used in REST2
            mol_charge (int, optional): Total charge of the ML zone. Defaults to 0.
            scaling_charges (float, optional): Scaling factor for MM zone charges,
              used for calibration of ML-MM interactions. Defaults to 1.0
              Check the AMP publications for optimal value for a given solvent. Defaults to 1.0.
            scaling_factor_alchemical_coulomb (float, optional): Lambda value for alchemical ML-MM decoupling,
              scales the electrostatic interaction. Defaults to 1.0.
            rank (Union[int, None], optional): Process rank. Is used in REST2 simulation. Defaults to None.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.system = system
        self.topology = topology
        self.qm_zone = qm_zone
        self.charges_mm = charges_mm
        self.mm_zone_charges = mm_zone_charges
        self.params_path = params_path
        self.weights_path = weights_path
        self.device_ml = device_ml
        self.scaling_factor_node_potential = scaling_factor_node_potential
        self.scaling_factor_coulomb_qm = scaling_factor_coulomb_qm
        self.scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm
        self.scaling_factor_D4 = scaling_factor_D4
        self.scaling_factor_ZBL = scaling_factor_ZBL
        self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
        self.mol_charge = mol_charge
        self.scaling_charges = scaling_charges
        self.rank = rank
        self.logger = logger

    def _load_params(self) -> None:
        """Helper function to load the YAML file with AMP parameters and save it to the python dictionary"""

        self.PARAMETERS = load_parameters(self.params_path)

    def _init_AMP_model(self) -> ScriptModule:
        """Function to setup the TorchForce with AMP model

        Returns:
            ScriptModule: Configured TorchForce with AMP and scrippted with JIT
        """

        if self.rank is not None:
            torch.cuda.set_device(self.rank)

        model = AMP(self.PARAMETERS).to(self.device_ml)
        model.load_state_dict(
            torch.load(
                self.weights_path, map_location=self.device_ml, weights_only=True
            )
        )
        torch.jit.enable_onednn_fusion(True)
        model_scripted = torch.jit.script(model)

        if self.logger:
            message = """The AMP model has been initialized successfully!"""
            self.logger.info(message)

        force_module = ForceModule(
            amp=model_scripted,
            topology=self.topology,
            qm_zone=self.qm_zone,
            charges_mm=self.charges_mm,
            mm_zone_charges=self.mm_zone_charges,
            device=self.device_ml,
            mol_charge=self.mol_charge,
            scaling_charges=self.scaling_charges,
        )

        module = torch.jit.script(force_module).to(self.device_ml)

        torch_force = TorchForce(module)

        torch_force.setUsesPeriodicBoundaryConditions(True)

        torch_force.addGlobalParameter(
            "scaling_factor_node_potential", self.scaling_factor_node_potential
        )

        torch_force.addGlobalParameter(
            "scaling_factor_coulomb_qm", self.scaling_factor_coulomb_qm
        )

        torch_force.addGlobalParameter(
            "scaling_factor_coulomb_qmmm", self.scaling_factor_coulomb_qmmm
        )

        torch_force.addGlobalParameter("scaling_factor_D4", self.scaling_factor_D4)

        torch_force.addGlobalParameter("scaling_factor_ZBL", self.scaling_factor_ZBL)

        torch_force.addGlobalParameter(
            "scaling_factor_alchemical_coulomb", self.scaling_factor_alchemical_coulomb
        )

        if self.logger:
            message = f"""Running ML model on {self.device_ml} with single precision."""
            self.logger.info(message)

        return torch_force

    def modify_forces(self) -> mm.System:
        """Function that adds AMP model as a TorchForce to the OpenMM System (the name of the added force is "AMP").

        Returns:
            mm.System: Updated OpenMM System with AMP model as a TorchForce
        """

        self._load_params()
        torch_force = self._init_AMP_model()
        torch_force.setName("AMP")
        self.system.addForce(torch_force)

        return self.system


class AmpConfigurator:
    def __init__(
        self,
        system: mm.System,
        topology: mm.app.topology.Topology,
        qm_zone_definition: Iterable[str],
        mm_zone_definition: Iterable[str],
        eps_rf: float,
        cutoff_nb: mm.unit.quantity.Quantity,
        params_path: str,
        weights_path: str,
        device_ml: str,
        scaling_factor_node_potential: float = 1.0,
        scaling_factor_coulomb_qm: float = 1.0,
        scaling_factor_coulomb_qmmm: float = 1.0,
        scaling_factor_D4: float = 1.0,
        scaling_factor_ZBL: float = 1.0,
        scaling_lj_qm_mm: float = 1.0,
        scaling_factor_alchemical_coulomb: float = 1.0,
        mol_charge: int = 0,
        scaling_charges: float = 1.0,
        tip4p: bool = False,
        softcore_lj_qm_mm: bool = False,
        rank: Union[int, None] = None,
        logger: Union[None, Logger] = None,
    ) -> None:
        """Constructor

        Args:
            system (mm.System): valid OpenMM System to which the AMP force will be added
            topology (mm.app.topology.Topology): Valid OpenMM Topology for the System of interest
            qm_zone_definition (Iterable[str]): Collection of residue names that are part of the ML zone
            mm_zone_definition (Iterable[str]): Collection of residue names that are part of the MM zone
            eps_rf (float): Dielectric permittivity of the medium, used for Reaction Field, unitless
            cutoff_nb (mm.unit.quantity.Quantity): The nonbonded cutoff value in [nm]
            params_path (str): Path to the YAML file with AMP configuration
            weights_path (str): Path to the AMP weights file
            device_ml (str): Device to be used for AMP model, corresponds to PyTorch device.
            scaling_factor_node_potential (float): Scaling factor for node potential, used in REST2
            scaling_factor_coulomb_qm (float): Scaling factor for ML-ML electrostatic interaction, used in REST2
            scaling_factor_coulomb_qmmm (float): Scaling factor for ML-MM electrostatic interaction, used in REST2
            scaling_factor_D4 (float): Scaling factor dispersion potential (ML-ML interaction, D4 potential), used in REST2
            scaling_factor_ZBL (float): Scaling factor ZBL potential (ML-ML interaction), used in REST2
            scaling_lj_qm_mm (float, optional): Lambda for the ML-MM Lennard-Jones interactions
            scaling in alchemical decoupling. Defaults to 1.0.
            scaling_factor_alchemical_coulomb (float, optional): Lambda value for alchemical ML-MM decoupling,
              scales the electrostatic interaction. Defaults to 1.0.
            mol_charge (int, optional): Total charge of the ML zone. Defaults to 0.
            scaling_charges (float, optional): Scaling factor for MM zone charges,
              used for calibration of ML-MM interactions. Defaults to 1.0.
            tip4p (bool, optional): Set True if the MM zone contains TIP4P molecules. Defaults to False.
            softcore_lj_qm_mm (bool, optional): Set True if the usual Lennard-Jones potential for ML-MM interaction
            should be substituted with softcore version in alchemical decoupling. Defaults to False.
            rank (Union[int, None], optional): Process rank. Is used in REST2 simulation. Defaults to None.
            logger (Union[None, Logger], optional): Logger. Defaults to None.
        """

        self.mm_zone_definition = mm_zone_definition
        self.qm_zone_definition = qm_zone_definition
        self.topology = topology
        self.system = system
        self.eps_rf = eps_rf
        self.cutoff_nb = cutoff_nb
        self.params_path = params_path
        self.weights_path = weights_path
        self.device_ml = device_ml
        self.scaling_factor_node_potential = scaling_factor_node_potential
        self.scaling_factor_coulomb_qm = scaling_factor_coulomb_qm
        self.scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm
        self.scaling_factor_D4 = scaling_factor_D4
        self.scaling_factor_ZBL = scaling_factor_ZBL
        self.scaling_lj_qm_mm = scaling_lj_qm_mm
        self.scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb
        self.mol_charge = mol_charge
        self.scaling_charges = scaling_charges
        self.tip4p = tip4p
        self.softcore_lj_qm_mm = softcore_lj_qm_mm
        self.rank = rank
        self.logger = logger

    def _define_zones(self) -> None:
        """Helper function to define the ML and MM zones based on residue names"""

        qm_zone, mm_zone = [], []

        if all([isinstance(i, str) for i in self.qm_zone_definition]) and all(
            [isinstance(i, str) for i in self.mm_zone_definition]
        ):
            qm_data = get_atom_indices(self.qm_zone_definition, self.topology)

            mm_data = get_atom_indices(self.mm_zone_definition, self.topology)

            for key in qm_data.keys():
                if qm_zone is None:
                    qm_zone = qm_data[key]

                else:
                    qm_zone.extend(qm_data[key])

            for key in mm_data.keys():
                if mm_zone is None:
                    mm_zone = mm_data[key]

                else:
                    mm_zone.extend(mm_data[key])

        self.qm_zone = np.array(sorted(qm_zone))
        self.mm_zone = np.array(sorted(mm_zone))

        if not self.tip4p:
            self.mm_zone_charges = np.array(sorted(mm_zone))
        else:
            tmp = []

            for atom in self.topology.atoms():
                if (atom.index in self.mm_zone) and (atom.name in ["H1", "H2", "M"]):
                    tmp.append(atom.index)

            self.mm_zone_charges = np.array(sorted(tmp))

        n_mm = len(self.mm_zone)
        n_qm = len(self.qm_zone)

        if self.logger:
            message = f"""Initialized systems with {n_qm} atoms in the QM zone and {n_mm} atoms in the MM zone."""

            self.logger.info(message)

    def _extract_mm_charges(self) -> None:
        """Helper function to collect the charges of the MM zone atoms"""

        for idf, force in enumerate(self.system.getForces()):
            if isinstance(force, mm.openmm.NonbondedForce) or isinstance(
                force, mm.openmm.CustomNonbondedForce
            ):
                nb_force = force

        self.charges_mm = np.array(
            [
                nb_force.getParticleParameters(index)[0]._value
                for index in self.mm_zone_charges
            ]
        )

    def _modify_old_forces(self) -> None:
        """Helper function to adjust all initially present classical forces to be used with AMP model for ML zone description"""

        force_modifier = AmpForcesModifier(
            system=self.system,
            topology=self.topology,
            qm_zone=self.qm_zone,
            mm_zone=self.mm_zone,
            eps_rf=self.eps_rf,
            cutoff_nb=self.cutoff_nb,
            tip4p=self.tip4p,
            scaling_lj_qm_mm=self.scaling_lj_qm_mm,
            softcore_lj_qm_mm=self.softcore_lj_qm_mm,
            logger=self.logger,
        )

        self.system = force_modifier.modify_forces()

    def _add_torch_force(self) -> None:
        """Helper function that adds the AMP model as TorchForce to the OpenMM System"""

        amp_force_adder = AmpTorchForceAdder(
            system=self.system,
            topology=self.topology,
            qm_zone=self.qm_zone,
            charges_mm=self.charges_mm,
            mm_zone_charges=self.mm_zone_charges,
            params_path=self.params_path,
            weights_path=self.weights_path,
            device_ml=self.device_ml,
            scaling_factor_node_potential=self.scaling_factor_node_potential,
            scaling_factor_coulomb_qm=self.scaling_factor_coulomb_qm,
            scaling_factor_coulomb_qmmm=self.scaling_factor_coulomb_qmmm,
            scaling_factor_D4=self.scaling_factor_D4,
            scaling_factor_ZBL=self.scaling_factor_ZBL,
            scaling_factor_alchemical_coulomb=self.scaling_factor_alchemical_coulomb,
            mol_charge=self.mol_charge,
            scaling_charges=self.scaling_charges,
            rank=self.rank,
            logger=self.logger,
        )

        self.system = amp_force_adder.modify_forces()

    def configure(self) -> mm.System:
        """Function that configures AMP model and adds it to the OpenMM System

        Returns:
            mm.System: Updated OpenMM System with AMP model as TorchForce
        """

        self._define_zones()
        self._extract_mm_charges()
        self._modify_old_forces()
        self._add_torch_force()

        return self.system


class ForceModule(torch.nn.Module):
    def __init__(
        self,
        amp: ScriptModule,
        topology: mm.app.topology.Topology,
        qm_zone: npt.NDArray[np.int_],
        charges_mm: npt.NDArray[np.floating],
        mm_zone_charges: npt.NDArray[np.int_],
        mol_charge: int = 0,
        n_nlist: int = 64,
        pairlist_padding: float = 4.0,
        chunk_size: int = 10000000,
        block_size: int = 4000,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        scaling_charges: float = 1.0,
    ):
        """Constructor

        Args:
            amp (ScriptModule): AMP model JIT-scrippted
            topology (mm.app.topology.Topology): Valid OpenMM topology
            qm_zone (npt.NDArray[np.int_]): QM zone atomic indices (0-based)
            charges_mm (npt.NDArray[np.floating]): Charges of the atoms in MM zone
            mm_zone_charges (npt.NDArray[np.int_]): Indices of the MM zone atoms with non-zero charges
            mol_charge (int, optional): Total charge of the QM zone. Defaults to 0.
            n_nlist (int, optional): Frequency of the neighborlist update in [steps]. Defaults to 64.
            pairlist_padding (float, optional): Distance padding in neighborlist creation in [Angstrom]. Defaults to 4.0.
            chunk_size (int, optional): Maximum size of the chunk for ML-ML neighborlist creation. Defaults to 10000000.
            block_size (int, optional): The number of ML and MM atoms processed simultaneously when building the ML-MM neighbor list. Defaults to 4000.
            dtype (torch.dtype, optional): Datatype to use for floats. Defaults to torch.float32.
            device (torch.device, optional): PyTorch device to use for the AMP model. Defaults to torch.device('cuda').
            scaling_charges (float, optional): Scaling factor for MM zone charges,
              used for calibration of ML-MM interactions. Defaults to 1.0.
        """

        super(ForceModule, self).__init__()
        self.topology = topology
        self.device = device
        self.dtype = dtype
        Z = [
            atom.element.atomic_number
            for atom in self.topology.atoms()
            if atom.index in qm_zone
        ]
        self.Z = torch.tensor(Z, device=self.device)
        self.register_buffer("nodes", amp.node_embedding(self.Z).detach())
        self.register_buffer(
            "charges_mm",
            torch.tensor(
                scaling_charges * charges_mm, device=self.device, dtype=self.dtype
            ).unsqueeze(-1),
        )
        self.register_buffer(
            "mol_charge", torch.tensor(mol_charge, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "mol_size",
            torch.tensor([self.Z.shape[0]], device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "mm_zone_charges", torch.tensor(mm_zone_charges, device=self.device)
        )
        self.register_buffer("qm_zone", torch.tensor(qm_zone, device=self.device))
        self.n_qm = self.qm_zone.shape[0]
        self.n_charges = self.mm_zone_charges.shape[0]
        self.register_buffer(
            "cutoff", torch.tensor(amp.cutoff, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "cutoff_esp",
            torch.tensor(amp.cutoff_esp, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_qmmm_esp",
            torch.tensor(amp.cutoff_qmmm_esp, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_qmmm_pol",
            torch.tensor(amp.cutoff_qmmm_pol, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_nlist",
            torch.tensor(
                amp.cutoff_esp + pairlist_padding, device=self.device, dtype=self.dtype
            ),
        )
        self.register_buffer(
            "cutoff_qmmm_nlist",
            torch.tensor(
                amp.cutoff_qmmm_esp + pairlist_padding,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.register_buffer(
            "chunk_size",
            torch.tensor(chunk_size, device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_block_size",
            torch.tensor(block_size, device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "step_count", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "n_nlist", torch.tensor(n_nlist, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_qm", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_mm", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_senders", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_receivers", torch.tensor(0, device=self.device, dtype=torch.int64)
        )

        self.amp = amp.to(device=device, dtype=dtype).eval()

    def forward(
        self,
        positions: torch.Tensor,
        boxvectors: torch.Tensor,
        scaling_factor_node_potential: torch.Tensor,
        scaling_factor_coulomb_qm: torch.Tensor,
        scaling_factor_coulomb_qmmm: torch.Tensor,
        scaling_factor_D4: torch.Tensor,
        scaling_factor_ZBL: torch.Tensor,
        scaling_factor_alchemical_coulomb: torch.Tensor,
    ) -> torch.Tensor:
        """Function that computes the potential energy of the system with AMP model

        Args:
            positions (torch.Tensor): positions[i,k] is the position (in nanometers) of spatial dimension k of particle i
            boxvectors (torch.Tensor): boxvectors[i,k] is the box vector component k (in nanometers) of box vector i
            scaling_factor_node_potential (torch.Tensor): Scaling factor used for node-potential scaling, used in REST2
            scaling_factor_coulomb_qm (torch.Tensor): Scaling factor for ML-ML electrostatic interaction, used in REST2
            scaling_factor_coulomb_qmmm (torch.Tensor): Scaling factor for ML-MM electrostatic interaction, used in REST2
            scaling_factor_D4 (torch.Tensor): Scaling factor dispersion potential (ML-ML interaction, D4 potential), used in REST2
            scaling_factor_ZBL (torch.Tensor): Scaling factor ZBL potential (ML-ML interaction), used in REST2
            scaling_factor_alchemical_coulomb (torch.Tensor): Lambda value for alchemical ML-MM decoupling,
              scales the electrostatic interaction.

        Returns:
            torch.Tensor: Potential energy associated with AMP model
        """

        scaling_factor_node_potential = scaling_factor_node_potential.to(
            dtype=self.dtype, device=self.device
        )
        scaling_factor_coulomb_qm = scaling_factor_coulomb_qm.to(
            dtype=self.dtype, device=self.device
        )
        scaling_factor_coulomb_qmmm = scaling_factor_coulomb_qmmm.to(
            dtype=self.dtype, device=self.device
        )
        scaling_factor_D4 = scaling_factor_D4.to(dtype=self.dtype, device=self.device)
        scaling_factor_ZBL = scaling_factor_ZBL.to(dtype=self.dtype, device=self.device)
        scaling_factor_alchemical_coulomb = scaling_factor_alchemical_coulomb.to(
            dtype=self.dtype, device=self.device
        )

        boxsize = (
            torch.diag(boxvectors * 10)
            .unsqueeze(0)
            .to(dtype=self.dtype, device=self.device)
        )  #
        positions = (positions * 10).to(dtype=self.dtype, device=self.device)
        graph = self._build_graph(positions, boxsize, scaling_factor_alchemical_coulomb)
        graph = self.amp(graph)
        node_potential = graph.V_nodes.squeeze() * scaling_factor_node_potential
        qm_coulomb = graph.V_coulomb_qm.squeeze() * scaling_factor_coulomb_qm
        qmmm_coulomb = graph.V_coulomb_qmmm.squeeze() * scaling_factor_coulomb_qmmm
        D4_potential = graph.V_D4.squeeze() * scaling_factor_D4
        ZBL_potential = graph.V_ZBL.squeeze() * scaling_factor_ZBL
        self.step_count = self.step_count + 1
        return node_potential + qm_coulomb + qmmm_coulomb + D4_potential + ZBL_potential

    def _build_graph(
        self,
        positions: torch.Tensor,
        boxsize: torch.Tensor,
        scaling_factor_alchemical_coulomb: torch.Tensor,
    ) -> Graph:
        """Helper function to create the input graph for AMP model from OpenMM data

        Args:
            positions (torch.Tensor): positions[i,k] is the position (in Angstrom) of spatial dimension k of particle i
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            scaling_factor_alchemical_coulomb (torch.Tensor): Lambda value for alchemical ML-MM decoupling,
              scales the electrostatic interaction.

        Returns:
            Graph: Input for the AMP model
        """
        coords_qm = positions[self.qm_zone]
        coords_mm = positions[self.mm_zone_charges]
        if self.step_count % self.n_nlist == 0:
            self.nlist_qm, self.nlist_mm = self.build_nlist_qmmm_iteratively(
                coords_qm, coords_mm, boxsize
            )
            trius = torch.triu_indices(
                self.n_qm, self.n_qm, offset=1, dtype=torch.long, device=self.device
            )
            senders_qm, receivers_qm = trius[0], trius[1]
            self.nlist_senders, self.nlist_receivers = self.build_nlist(
                coords_qm, coords_qm, boxsize, senders_qm, receivers_qm
            )
        R1_qm, Rx1_qm, senders_qm, receivers_qm = self.prepare_distances_qm(
            coords_qm, boxsize, self.nlist_senders, self.nlist_receivers
        )
        R1, R2, Rx1, Rx2, senders, receivers = self.prepare_qm_indices(
            R1_qm, Rx1_qm, senders_qm, receivers_qm
        )
        R1_esp, R2_esp, _, _, senders_esp, receivers_esp = self.prepare_esp_indices(
            R1_qm, Rx1_qm, senders_qm, receivers_qm
        )
        R1_qmmm, Rx1_qmmm, indices_qm, indices_mm = self.prepare_distances_qmmm(
            coords_qm, coords_mm, boxsize, self.nlist_qm, self.nlist_mm
        )
        (
            R1_qmmm_esp,
            Rx1_qmmm_esp,
            Rx2_qmmm_esp,
            indices_qm_esp,
            indices_mm_esp,
            R1_qmmm_pol,
            Rx1_qmmm_pol,
            Rx2_qmmm_pol,
            indices_qm_pol,
            indices_mm_pol,
        ) = self.prepare_qmmm_indices(R1_qmmm, Rx1_qmmm, indices_qm, indices_mm)
        mm_monos_esp, mm_monos_pol = (
            self.charges_mm[indices_mm_esp] * scaling_factor_alchemical_coulomb,
            self.charges_mm[indices_mm_pol] * scaling_factor_alchemical_coulomb,
        )
        graph = Graph(
            Z=self.Z,
            nodes=self.nodes,
            coords_qm=coords_qm,
            mm_monos_esp=mm_monos_esp,
            mm_monos_pol=mm_monos_pol,
            mol_charge=self.mol_charge,
            mol_size=self.mol_size,
            R1=R1,
            R2=R2,
            Rx1=Rx1,
            Rx2=Rx2,
            senders=senders,
            receivers=receivers,
            R1_esp=R1_esp,
            R2_esp=R2_esp,
            senders_esp=senders_esp,
            receivers_esp=receivers_esp,
            batch_index_esp=torch.empty(0),
            R1_qmmm_esp=R1_qmmm_esp,
            Rx1_qmmm_esp=Rx1_qmmm_esp,
            Rx2_qmmm_esp=Rx2_qmmm_esp,
            receivers_qmmm_esp=indices_qm_esp,
            qm_indices_qmmm_esp=torch.empty(0),
            R1_qmmm_pol=R1_qmmm_pol,
            Rx1_qmmm_pol=Rx1_qmmm_pol,
            Rx2_qmmm_pol=Rx2_qmmm_pol,
            receivers_qmmm_pol=indices_qm_pol,
            md_mode=True,
            n_channels=self.amp.n_channels,
        )
        return graph

    def prepare_distances_qm(
        self,
        coords_qm: torch.Tensor,
        boxsize: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function computing the ML-ML distances and ML-ML distance vectors with PBC for all pairs of ML-ML neighborlist

        Args:
            coords_qm (torch.Tensor): Initial coordinates of ML zone
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            senders (torch.Tensor): ML-ML Neighborlist of ML zone senders
            receivers (torch.Tensor): ML-ML Neighborlist of ML zone receivers

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ML-ML distances for sender-receiver pairs
            - ML-ML distance vectors for sender-receiver pairs
            - ML-ML Neighborlist of ML zone senders
            - ML-ML Neighborlist of ML zone receivers
        """
        R1_qm, Rx1_qm = ForceModuleUtilities.min_image(
            coords_qm, boxsize, senders, receivers
        )
        return R1_qm, Rx1_qm, senders, receivers

    def prepare_distances_qmmm(
        self,
        coords_qm: torch.Tensor,
        coords_mm: torch.Tensor,
        boxsize: torch.Tensor,
        indices_qm: torch.Tensor,
        indices_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper function computing the ML-MM distances and ML-MM distance vectors with PBC for all pairs of ML-MM neighborlist

        Args:
            coords_qm (torch.Tensor): Initial coordinates of ML zone
            coords_mm (torch.Tensor): Initial coordinates of MM zone
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            indices_qm (torch.Tensor): ML-MM Neighborlist of ML zone receivers
            indices_mm (torch.Tensor): ML-MM Neighborlist of MM zone senders

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ML-MM distances for sender-receiver pairs
            - ML-MM distance vectors for sender-receiver pairs
            - ML-MM Neighborlist of ML zone receivers
            - ML-MM Neighborlist of MM zone senders

        """
        R1_qmmm, Rx1_qmmm = ForceModuleUtilities.min_image_qmmm(
            coords_qm, coords_mm, boxsize, indices_qm, indices_mm
        )
        return R1_qmmm, Rx1_qmmm, indices_qm, indices_mm

    def prepare_qm_indices(
        self,
        R1_qm: torch.Tensor,
        Rx1_qm: torch.Tensor,
        senders_qm: torch.Tensor,
        receivers_qm: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Helper function to apply the ML-ML graph cutoff to the ML-ML sender-receiver pairs.
        Filters out all pairs that are further away than ML-ML graph cutoff (4 Angstrom) yielding the ML-ML sender-receiver indices.
        Filters respectively the ML-ML distances, ML-ML distance vectors.
        Computes the outer product of the ML-ML distance vectors with themselves.

        Args:
            R1_qm (torch.Tensor): ML-ML distances computed for all pairs of ML-ML neighborlist
            Rx1_qm (torch.Tensor): ML-ML distance vectors computed for all pairs of ML-ML neighborlist
            senders_qm (torch.Tensor): All possible ML zone senders
            receivers_qm (torch.Tensor): All possible ML zone receivers

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ML-ML distances within graph cutoff for sender-receiver pairs
            - ML-ML distances squared within graph cutoff for sender-receiver pairs
            - ML-ML distance vectors within graph cutoff for sender-receiver pairs
            - Self outer products of ML-ML distance vectors within graph cutoff for sender-receiver pairs
            - ML zone sender indices within graph cutoff
            - ML zone receiver indices within graph cutoff
        """
        cutoff_indices = torch.where(R1_qm < self.amp.cutoff)[0]
        R1 = torch.index_select(R1_qm, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1_qm, dim=0, index=cutoff_indices)
        senders_qm = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers_qm = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R1 = torch.cat((R1, R1))
        R2 = torch.square(R1)
        Rx1 = torch.cat((Rx1, -Rx1), dim=0) / R1
        Rx2 = build_Rx2(Rx1)
        Rx1, Rx2 = Rx1.unsqueeze(1), Rx2.unsqueeze(1)
        senders = torch.cat((senders_qm, receivers_qm))
        receivers = torch.cat((receivers_qm, senders_qm))
        return R1, R2, Rx1, Rx2, senders, receivers

    def prepare_esp_indices(
        self,
        R1: torch.Tensor,
        Rx1: torch.Tensor,
        senders_qm: torch.Tensor,
        receivers_qm: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Helper function to apply the ML-ML electrostatic interaction graph cutoff (TODO in paper) to the ML-ML sender-receiver pairs.
        Filters out all pairs that are further away than ML-ML electrostatic interaction graph cutoff (14 Angstrom) yielding
        the ML-ML sender-receiver indices that participate in electrostatic interaction.
        Filters respectively the ML-ML distances, ML-ML distance vectors.
        Computes the outer product of the ML-ML distance vectors with themselves.

        Args:
            R1_qm (torch.Tensor): ML-ML distances computed for all pairs of ML-ML neighborlist
            Rx1_qm (torch.Tensor): ML-ML distance vectors computed for all pairs of ML-ML neighborlist
            senders_qm (torch.Tensor): All possible ML zone senders
            receivers_qm (torch.Tensor): All possible ML zone receivers

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ML-ML distances within ML-ML electrostatic interaction graph cutoff for sender-receiver pairs
            - ML-ML distances squared within ML-ML electrostatic interaction graph cutoff for sender-receiver pairs
            - ML-ML distance vectors within ML-ML electrostatic interaction graph cutoff for sender-receiver pairs
            - Self outer products of ML-ML distance vectors within ML-ML electrostatic interaction graph cutoff for sender-receiver pairs
            - ML zone sender indices within ML-ML electrostatic interaction graph cutoff
            - ML zone receiver indices within ML-ML electrostatic interaction graph cutoff
        """
        cutoff_indices = torch.where(R1 < self.amp.cutoff_esp)[0]
        R1 = torch.index_select(R1, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1, dim=0, index=cutoff_indices)
        senders = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R2 = torch.square(R1)
        Rx2 = build_Rx2(Rx1)
        return R1, R2, Rx1, Rx2, senders, receivers

    def prepare_qmmm_indices(
        self,
        R1: torch.Tensor,
        Rx1: torch.Tensor,
        indices_qm: torch.Tensor,
        indices_mm: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Helper function to apply the ML-MM graph cutoffs (ML/MM polarization and ML/MM electrostatics)
        to the ML-ML sender-receiver pairs. Filters out all pairs that are further away than ML-MM graph
        cutoffs (8 and 9 Angstrom for polarization and electrostatics respectively)
        yielding the ML-MM sender-receiver indices for both types of interactions.
        Filters respectively the ML-MM distances, ML-MM distance vectors.
        Computes the outer product of the ML-MM distance vectors with themselves.

        Args:
            R1 (torch.Tensor): ML-MM distances computed for all pairs of ML-MM neighborlist
            Rx1 (torch.Tensor):  ML-MM distance vectors computed for all pairs of ML-MM neighborlist
            indices_qm (torch.Tensor):  ML-MM Neighborlist of ML zone receivers
            indices_mm (torch.Tensor): ML-MM Neighborlist of MM zone senders

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - ML-MM distances within ML/MM electrostatics graph cutoff for sender-receiver pairs
            - ML-MM distance vectors within  ML/MM electrostatics graph cutoff for sender-receiver pairs
            - Self outer products of ML-MM distance vectors within ML/MM electrostatics graph cutoff for sender-receiver pairs
            - ML zone receiver indices within ML/MM electrostatics graph cutoff
            - MM zone sender indices within ML/MM electrostatics graph cutoff
            - ML-MM distances within ML/MM polarization graph cutoff for sender-receiver pairs
            - ML-MM distance vectors within  ML/MM polarization graph cutoff for sender-receiver pairs
            - Self outer products of ML-MM distance vectors within ML/MM polarization graph cutoff for sender-receiver pairs
            - ML zone receiver indices within ML/MM polarization graph cutoff
            - MM zone sender indices within ML/MM polarization graph cutoff
        """
        cutoff_indices_esp = torch.where(R1 < self.amp.cutoff_qmmm_esp)[0]
        R1_qmmm_esp = torch.index_select(R1, dim=0, index=cutoff_indices_esp)
        Rx1_qmmm_esp = torch.index_select(Rx1, dim=0, index=cutoff_indices_esp)
        Rx2_qmmm_esp = build_Rx2(Rx1_qmmm_esp)
        indices_qm_esp = torch.index_select(indices_qm, dim=0, index=cutoff_indices_esp)
        indices_mm_esp = torch.index_select(indices_mm, dim=0, index=cutoff_indices_esp)
        cutoff_indices_pol = torch.where(R1_qmmm_esp < self.amp.cutoff_qmmm_pol)[0]
        R1_qmmm_pol = torch.index_select(R1_qmmm_esp, dim=0, index=cutoff_indices_pol)
        Rx1_qmmm_pol = (
            torch.index_select(Rx1_qmmm_esp, dim=0, index=cutoff_indices_pol)
            / R1_qmmm_pol
        )
        Rx2_qmmm_pol = build_Rx2(Rx1_qmmm_pol)
        indices_qm_pol = torch.index_select(
            indices_qm_esp, dim=0, index=cutoff_indices_pol
        )
        indices_mm_pol = torch.index_select(
            indices_mm_esp, dim=0, index=cutoff_indices_pol
        )
        return (
            R1_qmmm_esp,
            Rx1_qmmm_esp,
            Rx2_qmmm_esp,
            indices_qm_esp,
            indices_mm_esp,
            R1_qmmm_pol,
            Rx1_qmmm_pol,
            Rx2_qmmm_pol,
            indices_qm_pol,
            indices_mm_pol,
        )

    def build_nlist_qmmm_iteratively(
        self,
        positions_a: torch.Tensor,
        positions_b: torch.Tensor,
        boxsize: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to build the ML/MM neighborlist iteratively

        Args:
            positions_a (torch.Tensor): Positions of ML zone atoms
            positions_b (torch.Tensor): Positions of MM zone atoms
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Neighborlist for ML zone atoms
            - Neighborlist for MM zone atoms
        """
        with torch.no_grad():
            nlist_a, nlist_b = [], []
            a, b = (
                min(self.n_qm, self.index_block_size),
                min(self.n_charges, self.index_block_size),
            )
            index_matrix = torch.full((a, b), 1, device=self.device, dtype=torch.bool)
            block_indices_qm = torch.arange(0, self.n_qm, self.index_block_size)
            block_indices_mm = torch.arange(0, self.n_charges, self.index_block_size)
            for offset_index_qm in block_indices_qm:
                end_index_qm = self.index_block_size
                if (offset_index_qm + self.index_block_size) > self.n_qm:
                    end_index_qm = self.n_qm % self.index_block_size
                for offset_index_mm in block_indices_mm:
                    end_index_mm = self.index_block_size
                    if (offset_index_mm + self.index_block_size) > self.n_charges:
                        end_index_mm = self.n_charges % self.index_block_size
                    indices_qm, indices_mm = torch.where(
                        index_matrix[:end_index_qm, :end_index_mm]
                    )
                    indices_qm = indices_qm + offset_index_qm
                    indices_mm = indices_mm + offset_index_mm
                    R1, Rx1 = ForceModuleUtilities.min_image_block(
                        positions_a[indices_qm], positions_b[indices_mm], boxsize
                    )
                    cutoff_indices = torch.where(R1.squeeze() < self.cutoff_qmmm_nlist)[
                        0
                    ]
                    indices_qm = torch.index_select(
                        indices_qm, dim=0, index=cutoff_indices
                    )
                    indices_mm = torch.index_select(
                        indices_mm, dim=0, index=cutoff_indices
                    )
                    nlist_a.append(indices_qm)
                    nlist_b.append(indices_mm)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)

    def build_nlist(
        self,
        positions_a: torch.Tensor,
        positions_b: torch.Tensor,
        boxsize: torch.Tensor,
        indices_a: torch.Tensor,
        indices_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to build the ML/ML neighborlist

        Args:
            positions_a (torch.Tensor): Positions of ML zone atoms
            positions_b (torch.Tensor): Positions of ML zone atoms
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            indices_a (torch.Tensor): ML zone senders indices
            indices_b (torch.Tensor): ML zone receivers indices

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Nieghborlist for ML zone senders
            - Nieghborlist for ML zone receivers
        """
        with torch.no_grad():
            chunks_a, chunks_b = (
                ForceModuleUtilities.chunkify(indices_a, self.chunk_size),
                ForceModuleUtilities.chunkify(indices_b, self.chunk_size),
            )
            nlist_a, nlist_b = [], []
            for chunk_a, chunk_b in zip(chunks_a, chunks_b):
                R1, _ = ForceModuleUtilities.min_image_block(
                    positions_a[chunk_a], positions_b[chunk_b], boxsize
                )
                cutoff_indices = torch.where(R1.squeeze() < self.cutoff_nlist)[0]
                chunk_a_nlist = torch.index_select(chunk_a, dim=0, index=cutoff_indices)
                chunk_b_nlist = torch.index_select(chunk_b, dim=0, index=cutoff_indices)
                nlist_a.append(chunk_a_nlist)
                nlist_b.append(chunk_b_nlist)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)


class ForceModuleUtilities:
    def __init__(self):
        pass

    # Assuming orthorombic box
    @staticmethod
    def to_fractional(coords: torch.Tensor, boxsize: torch.Tensor) -> torch.Tensor:
        """Helper function to transform coordinates to fractional coordinates

        Args:
            coords (torch.Tensor): Initial coordinates
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)

        Returns:
            torch.Tensor: Positions in fractional coordinates
        """
        return coords / boxsize

    @staticmethod
    def from_fractional(coords: torch.Tensor, boxsize: torch.Tensor) -> torch.Tensor:
        """Helper function to transform coordinates from fractional coordinates to normal

        Args:
            coords (torch.Tensor): Positions in fractional coordinates
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)

        Returns:
            torch.Tensor: Positions in normal coordinates
        """
        return coords * boxsize

    @staticmethod
    def min_image(
        coords: torch.Tensor,
        boxsize: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to apply minimum image convention and compute the ML-ML distances under PBC

        Args:
            coords (torch.Tensor): Coordinates of ML zone atoms
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            senders (torch.Tensor): Indices of ML zone atoms that are senders
            receivers (torch.Tensor): Indices of ML zone atoms that are receivers

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Distances between ML zone senders and receivers under PBC
            - Distance vectors between ML zone senders and receivers under PBC
        """
        coords_a, coords_b = coords[senders], coords[receivers]
        return ForceModuleUtilities.min_image_block(coords_a, coords_b, boxsize)

    @staticmethod
    def min_image_qmmm(
        coords_qm: torch.Tensor,
        coords_mm: torch.Tensor,
        boxsize: torch.Tensor,
        indices_qm: torch.Tensor,
        indices_mm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to apply minimum image convention and compute the ML-MM distances under PBC

        Args:
            coords_qm (torch.Tensor): Coordinates of ML zone atoms
            coords_mm (torch.Tensor): Coordinates of MM zone atoms
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)
            indices_qm (torch.Tensor): Indices of ML zone atoms that are receivers
            indices_mm (torch.Tensor): Indices of MM zone atoms that are senders

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Distances between ML zone receivers and MM zone senders under PBC
            - Distance vectors between ML zone receivers and MM zone senders under PBC
        """
        coords_a, coords_b = coords_qm[indices_qm], coords_mm[indices_mm]
        return ForceModuleUtilities.min_image_block(coords_a, coords_b, boxsize)

    @staticmethod
    def min_image_block(
        coords_a: torch.Tensor, coords_b: torch.Tensor, boxsize: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to compute the distance and distance vectors between sets of particles under PBC

        Args:
            coords_a (torch.Tensor): Coordinates of set A
            coords_b (torch.Tensor): Coordinates of set B
            boxsize (torch.Tensor): Tensor with box lengths [[Lx, Ly, Lz]] in each dimension (1, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - Distances between particles in sets A and B under PBC
            - Distance vectors between particles in sets A and B under PBC
        """
        Rx1 = coords_b - coords_a
        Rx1 = Rx1 - ForceModuleUtilities.from_fractional(
            torch.round(ForceModuleUtilities.to_fractional(Rx1, boxsize)), boxsize
        )
        R1 = torch.linalg.norm(Rx1, dim=-1, keepdim=True)
        return R1, Rx1

    @staticmethod
    def chunkify(indices: torch.Tensor, chunk_size: torch.Tensor) -> List[torch.Tensor]:
        """Wrapper of torch.split function

        Args:
            indices (torch.Tensor): Check the torch.split documentation
            chunk_size (torch.Tensor): Check the torch.split documentation

        Returns:
            List[torch.Tensor]: Check the torch.split documentation
        """
        return torch.split(indices, chunk_size)
