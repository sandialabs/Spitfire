"""
This is the base spitfire module directory.
"""

from .chemistry.library import Dimension, Library
from .chemistry.analysis import get_ct_solution_array
from .chemistry.mechanism import ChemicalMechanismSpec
from .chemistry.flamelet import Flamelet
from .chemistry.reactors import HomogeneousReactor
from .chemistry.tabulation import build_adiabatic_eq_library, \
    build_adiabatic_bs_library, \
    build_adiabatic_slfm_library, \
    build_unreacted_library, \
    build_nonadiabatic_defect_bs_library, \
    build_nonadiabatic_defect_eq_library, \
    build_nonadiabatic_defect_transient_slfm_library

from .griffon.griffon import PyCombustionKernels

from .time.integrator import SaveAllDataToList, odesolve
from .time.nonlinear import SimpleNewtonSolver
from .time.stepcontrol import PIController
from .time.methods import (ForwardEulerS1P1,
                           ExpMidpointS2P2,
                           ExpTrapezoidalS2P2Q1,
                           ExpRalstonS2P2,
                           RK3KuttaS3P3,
                           RK4ClassicalS4P4,
                           CashKarpS6P5Q4,
                           BackwardEulerS1P1Q1,
                           CrankNicolsonS2P2,
                           SDIRKS2P2,
                           KennedyCarpenterS6P4Q3,
                           GeneralAdaptiveERK,
                           GeneralAdaptiveERKMultipleEmbedded)
