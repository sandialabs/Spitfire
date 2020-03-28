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
from .time.methods import ESDIRK64, \
    SDIRK22, \
    BackwardEuler, \
    BackwardEulerWithError, \
    CrankNicolson, \
    ForwardEuler, \
    ExplicitRungeKutta2Midpoint, \
    ExplicitRungeKutta2Ralston, \
    ExplicitRungeKutta2Trapezoid, \
    ExplicitRungeKutta3Kutta, \
    ExplicitRungeKutta4Classical, \
    GeneralAdaptiveExplicitRungeKutta, \
    AdaptiveERK54CashKarp, \
    AdaptiveERK21HeunEuler
