""" "
Copyright (C) 2026 ETH Zurich, Igor Gordiy, and other AMP contributors
"""

from abc import abstractmethod
from enum import Enum

import openmm as mm


class NonbondedMethod(Enum):
    PME = mm.app.forcefield.PME
    NoCutoff = mm.app.forcefield.NoCutoff
    CutoffNonPeriodic = mm.app.forcefield.CutoffNonPeriodic
    CutoffPeriodic = mm.app.forcefield.CutoffPeriodic
    Ewald = mm.app.forcefield.Ewald
    LJPME = mm.app.forcefield.LJPME


class Constraints(Enum):
    None_ = None
    HBonds = mm.app.forcefield.HBonds


class ForcesModifier:
    @abstractmethod
    def modify_forces():
        pass
