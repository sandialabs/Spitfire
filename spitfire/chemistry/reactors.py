"""
This module contains the HomogeneousReactor class that provides a high-level interface for a variety of 0-D reactors
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

from spitfire.time.governor import Governor, CustomTermination, Steady, FinalTime, SaveAllDataToList
from spitfire.time.methods import ESDIRK64
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
from spitfire.chemistry.library import Dimension, Library
import numpy as np
from numpy import zeros, hstack, sqrt, sum
from scipy.linalg.lapack import dgetrf as lapack_lu_factor
from scipy.linalg.lapack import dgetrs as lapack_lu_solve
from scipy.sparse.linalg import splu as superlu_factor
from scipy.sparse import csc_matrix as sparse_csc_matrix
import matplotlib.pyplot as plt


class HomogeneousReactor(object):
    """A class for solving zero-dimensional reactors

    **Constructor**: specify a mechanism, initial mixture, and reactor specifications

    Parameters
    ----------
    mech_spec : spitfire.chemistry.mechanism.ChemicalMechanismSpec instance
        the mechanism
    initial_mixture : Cantera.Quantity (a Spitfire stream) or Cantera.gas object
        the initial mixture of the reactor
    configuration : str
        whether the reactor is constant-volume (isochoric) or constant-pressure (isobaric)
    heat_transfer : str
        whether the reactor is adiabatic, isothermal, or diathermal (finite-rate heat transfer by convection and/or radiation)
    mass_transfer : str
        whether the reactor is closed or open
    convection_temperature : float or callable
        the temperature of external fluid in a diathermal reactor, either a constant or a function of time, f(t)
    radiation_temperature : float or callable
        the temperature of external radiation body of a diathermal reactor, either a constant or a function of time, f(t)
    convection_coefficient : float or callable
        the convection coefficient, either a constant or a function of time, f(t)
    radiative_emissivity : float or callable
        the radiative emissivity, either a constant or a function of time, f(t)
    shape_dimension_dict : float or callable
        The shape and dimension of a diathermal reactor. The shape is one of 'cube', 'sphere', 'capsule', 'tetrahedron', 'octahedron', or 'icosahedron'
        (see `wikipedia <https://en.wikipedia.org/wiki/Surface-area-to-volume_ratio#Mathematical_examples>`_).
        The dimension is either the characteristic length ('char. length') in meters or volume ('volume') in cubic meters.
    mixing_tau : float or callable
        The mixing time of an open reactor, either a constant or a function of time, f(t)
    feed_temperature : float or callable
        the temperature of the feed stream of an open reactor, either a constant or a function of time, f(t)
    feed_mass_fractions : np.array or callable
        the mass fractions of the feed stream of an open reactor, either a constant or a function of time, f(t)
    feed_density : float or callable
        the density of the feed stream of an open reactor, either a constant or a function of time, f(t)
    rates_sensitivity_type : str
        how the chemical source term Jacobian is formed, either 'dense' or 'sparse' for exact formulations
        or 'no-TBAF' which ignores third-body and falloff sensitivities. The default is 'dense'.
        For large mechanisms (over 100 species) the 'sparse' formulation is far faster than 'dense',
        especially for mechanisms of more than 300 species.
    sensitivity_transform_type : str
        how the Jacobian is transformed for isobaric systems, currently only 'exact' is supported
    initial_time : float
        the starting time point (in seconds) of the reactor, default to 0.0
    """

    _configurations = ['constant pressure', 'constant volume', 'isobaric', 'isochoric']
    _configuration_dict = {'constant pressure': 'isobaric',
                           'isobaric': 'isobaric',
                           'constant volume': 'isochoric',
                           'isochoric': 'isochoric'}

    _heat_transfers = ['adiabatic', 'isothermal', 'diathermal']

    _mass_transfers = ['closed', 'open']

    _shape_dict = {'cube': {'l->sov': lambda a: 6. / a,
                            'v->sov': lambda v: 6. / (np.power(v, 1. / 3.))},
                   'sphere': {'l->sov': lambda a: 3. / a,
                              'v->sov': lambda v: 3. / (np.power(3. * v / (4. * np.pi), 1. / 3.))},
                   'capsule': {'l->sov': lambda a: 12. / (5. * a),
                               'v->sov': lambda v: 12. / (5. * np.power(3. * v / (10. * np.pi), 1. / 3.))},
                   'tetrahedron': {'l->sov': lambda a: 6. * sqrt(6.) / a,
                                   'v->sov': lambda v: 6. * sqrt(6.) / np.power(12. * v / np.sqrt(2.), 1. / 3.)},
                   'octahedron': {'l->sov': lambda a: 3. * sqrt(6.) / a,
                                  'v->sov': lambda v: 3. * sqrt(6.) / np.power(3. * v / np.sqrt(2.), 1. / 3.)},
                   'icosahedron': {'l->sov': lambda a: 12. * sqrt(3.) / ((3. + sqrt(5.)) * a),
                                   'v->sov': lambda v: 12. * sqrt(3.) / (
                                       (3. + sqrt(5.)) * np.power(12. * v / 5. / (3. + sqrt(5.)), 1. / 3.))}}
    _shapes = list(_shape_dict.keys())

    @classmethod
    def _check_constructor_argument(cls, argument, description, acceptable_values):
        if argument.lower() in acceptable_values:
            return True
        else:
            raise ValueError(
                """
                Error in reactor construction:
                    Bad {:} argument detected.
                    Argument given: {:}
                    Acceptable values: {:}
                """.format(description, argument, acceptable_values))

    @classmethod
    def _warn_unused_argument(cls, argument_should_be_none, unused_argument, reason):
        if argument_should_be_none is not None:
            print(
                """
                Warning in reactor construction:
                    The {:} argument is unused.
                    Reason: {:}
                """.format(unused_argument, reason))

    @classmethod
    def _check_necessary_argument(cls, argument_should_be_not_none, unspecified_argument, reason):
        if argument_should_be_not_none is None:
            raise ValueError(
                """
                Error in reactor construction:
                    The {:} argument is needed but was unspecified.
                    Reason: {:}
                """.format(unspecified_argument, reason))
        else:
            return True

    def __init__(self,
                 mech_spec,
                 initial_mixture,
                 configuration,
                 heat_transfer,
                 mass_transfer,
                 convection_temperature=None,
                 radiation_temperature=None,
                 convection_coefficient=None,
                 radiative_emissivity=None,
                 shape_dimension_dict=None,
                 mixing_tau=None,
                 feed_temperature=None,
                 feed_mass_fractions=None,
                 feed_density=None,
                 rates_sensitivity_type='dense',
                 sensitivity_transform_type='exact',
                 initial_time=0.):

        # check configuration (isobaric/isochoric), heat transfer, and mass transfer
        if self._check_constructor_argument(configuration, 'configuration', self._configurations):
            self._configuration = self._configuration_dict[configuration.lower()]

        if self._check_constructor_argument(heat_transfer, 'heat transfer', self._heat_transfers):
            self._heat_transfer = heat_transfer.lower()

        if self._check_constructor_argument(mass_transfer, 'mass transfer', self._mass_transfers):
            self._mass_transfer = mass_transfer.lower()

        # save heat transfer parameters, check validity, and warn for unused parameters
        if self._heat_transfer == 'adiabatic' or self._heat_transfer == 'isothermal':
            self._convection_temperature = 0.
            self._radiation_temperature = 0.
            self._convection_coefficient = 0.
            self._radiative_emissivity = 0.
            self._surface_area_to_volume = 0.

            message = 'heat transfer is not set to diathermal'
            self._warn_unused_argument(convection_temperature, 'convection_temperature', message)
            self._warn_unused_argument(radiation_temperature, 'radiation_temperature', message)
            self._warn_unused_argument(convection_coefficient, 'convection_coefficient', message)
            self._warn_unused_argument(radiative_emissivity, 'radiative_emissivity', message)
            self._warn_unused_argument(shape_dimension_dict, 'shape_dimension_dict', message)
        else:
            message = 'heat transfer is set to diathermal'
            if self._check_necessary_argument(convection_temperature, 'convection_temperature', message):
                self._convection_temperature = convection_temperature

            if self._check_necessary_argument(radiation_temperature, 'radiation_temperature', message):
                self._radiation_temperature = radiation_temperature

            if self._check_necessary_argument(convection_coefficient, 'convection_coefficient', message):
                self._convection_coefficient = convection_coefficient

            if self._check_necessary_argument(radiative_emissivity, 'radiative_emissivity', message):
                self._radiative_emissivity = radiative_emissivity

            if self._check_necessary_argument(shape_dimension_dict, 'shape_dimension_dict', message):
                if 'shape' not in shape_dimension_dict:
                    raise ValueError(
                        """
                        Error in reactor construction:
                            The shape_dimension_dict argument did not have the required ''shape'' key
                        """)
                else:
                    self._check_constructor_argument(shape_dimension_dict['shape'], 'shape', self._shapes)

                if 'char. length' not in shape_dimension_dict and 'volume' not in shape_dimension_dict:
                    raise ValueError(
                        """
                        Error in reactor construction:
                            The shape_dimension_dict argument did not have one of the required ''char. length'' or ''volume'' keys
                        """)
                elif 'char. length' in shape_dimension_dict and 'volume' in shape_dimension_dict:
                    raise ValueError(
                        """
                        Error in reactor construction:
                            The shape_dimension_dict argument had both of the ''char. length'' or ''volume'' keys. Only one is allowed.
                        """)

                if 'char. length' in shape_dimension_dict:
                    method = self._shape_dict[shape_dimension_dict['shape']]['l->sov']
                    self._surface_area_to_volume = method(shape_dimension_dict['char. length'])
                elif 'volume' in shape_dimension_dict:
                    method = self._shape_dict[shape_dimension_dict['shape']]['v->sov']
                    self._surface_area_to_volume = method(shape_dimension_dict['volume'])

        # save mass transfer specifics and check validity
        if self._mass_transfer == 'closed':
            self._mixing_tau = 0.
            self._feed_temperature = 0.
            self._feed_mass_fractions = np.ndarray(1)
            self._feed_density = 0.

            message = 'mass transfer is not set to open'
            self._warn_unused_argument(mixing_tau, 'mixing_tau', message)
            self._warn_unused_argument(feed_temperature, 'feed_temperature', message)
            self._warn_unused_argument(feed_mass_fractions, 'feed_mass_fractions', message)
            self._warn_unused_argument(feed_density, 'feed_density', message)
        else:
            message = 'mass transfer is set to open'
            if self._check_necessary_argument(mixing_tau, 'mixing_tau', message):
                self._mixing_tau = np.Inf if mixing_tau is None else mixing_tau

            if self._check_necessary_argument(feed_temperature, 'feed_temperature', message):
                self._feed_temperature = feed_temperature

            if self._check_necessary_argument(feed_mass_fractions, 'feed_mass_fractions', message):
                self._feed_mass_fractions = feed_mass_fractions

            if self._configuration == 'isobaric':
                self._warn_unused_argument(feed_density, 'feed_density',
                                           message + ' but the reactor is isobaric, so feed_density is not needed')
                self._feed_density = feed_density
            else:
                if self._check_necessary_argument(feed_density, 'feed_density', message):
                    self._feed_density = feed_density

        # look for parameters that are functions of time
        self._parameter_time_functions = set()
        for attribute in ['_convection_temperature',
                          '_radiation_temperature',
                          '_convection_coefficient',
                          '_radiative_emissivity',
                          '_mixing_tau',
                          '_feed_temperature',
                          '_feed_mass_fractions',
                          '_feed_density']:
            if callable(getattr(self, attribute)):
                self._parameter_time_functions.add(attribute)

        self._tc_is_timevar = '_convection_temperature' in self._parameter_time_functions
        self._tr_is_timevar = '_radiation_temperature' in self._parameter_time_functions
        self._cc_is_timevar = '_convection_coefficient' in self._parameter_time_functions
        self._re_is_timevar = '_radiative_emissivity' in self._parameter_time_functions
        self._tau_is_timevar = '_mixing_tau' in self._parameter_time_functions
        self._tf_is_timevar = '_feed_temperature' in self._parameter_time_functions
        self._yf_is_timevar = '_feed_mass_fractions' in self._parameter_time_functions
        self._rf_is_timevar = '_feed_density' in self._parameter_time_functions

        self._tc_value = self._convection_temperature(0.) if self._tc_is_timevar else self._convection_temperature
        self._cc_value = self._convection_coefficient(0.) if self._cc_is_timevar else self._convection_coefficient
        self._tr_value = self._radiation_temperature(0.) if self._tr_is_timevar else self._radiation_temperature
        self._re_value = self._radiative_emissivity(0.) if self._re_is_timevar else self._radiative_emissivity
        self._tau_value = self._mixing_tau(0.) if self._tau_is_timevar else self._mixing_tau
        self._tf_value = self._feed_temperature(0.) if self._tf_is_timevar else self._feed_temperature
        self._yf_value = self._feed_mass_fractions(0.) if self._yf_is_timevar else self._feed_mass_fractions
        self._rf_value = self._feed_density(0.) if self._rf_is_timevar else self._feed_density

        self._rates_sensitivity_option = {'dense': 0, 'no-TBAF': 1, 'sparse': 2}[rates_sensitivity_type]
        self._sensitivity_transform_option = {'exact': 0}[sensitivity_transform_type]
        self._is_open = self._mass_transfer == 'open'
        self._heat_transfer_option = {'adiabatic': 0, 'isothermal': 1, 'diathermal': 2}[self._heat_transfer]

        self._gas = mech_spec.copy_stream(initial_mixture)
        self._mechanism = mech_spec
        self._griffon = self._mechanism.griffon

        self._initial_pressure = initial_mixture.P
        self._current_pressure = np.copy(self._initial_pressure)

        self._initial_temperature = initial_mixture.T
        self._current_temperature = np.copy(self._initial_temperature)

        self._initial_mass_fractions = initial_mixture.Y
        self._current_mass_fractions = np.copy(self._initial_mass_fractions)

        self._initial_time = np.copy(initial_time)
        self._current_time = np.copy(self._initial_time)

        self._n_species = self._gas.n_species
        self._n_reactions = self._gas.n_reactions
        self._n_equations = self._n_species if self._configuration == 'isobaric' else self._n_species + 1

        self._initial_state = zeros(self._n_equations)
        if self._configuration == 'isobaric':
            self._temperature_index = 0
            self._initial_state[0] = np.copy(initial_mixture.T)
            self._initial_state[1:] = np.copy(initial_mixture.Y[:-1])
        elif self._configuration == 'isochoric':
            self._temperature_index = 1
            self._initial_state[0] = np.copy(initial_mixture.density)
            self._initial_state[1] = np.copy(initial_mixture.T)
            self._initial_state[2:] = np.copy(initial_mixture.Y[:-1])

        self._current_state = np.copy(self._initial_state)

        self._variable_scales = np.ones(self._n_equations)
        self._variable_scales[self._temperature_index] = 1.e3
        self._left_hand_side_inverse_operator = None
        self._diag_indices = np.diag_indices(self._n_equations)

        self._extra_logger_title_line1 = f'{"":<10} | {"":<10}|'
        self._extra_logger_title_line2 = f'  {"T (K)":<8} | {"T-T_0 (K)":<10}|'

    def _update_heat_transfer_parameters(self, t):
        self._tc_value = self._convection_temperature(t) if self._tc_is_timevar else self._tc_value
        self._cc_value = self._convection_coefficient(t) if self._cc_is_timevar else self._cc_value
        self._tr_value = self._radiation_temperature(t) if self._tr_is_timevar else self._tr_value
        self._re_value = self._radiative_emissivity(t) if self._re_is_timevar else self._re_value

    def _update_mass_transfer_parameters(self, t):
        self._tau_value = self._mixing_tau(t) if self._tau_is_timevar else self._tau_value
        self._tf_value = self._feed_temperature(t) if self._tf_is_timevar else self._tf_value
        self._yf_value = self._feed_mass_fractions(t) if self._yf_is_timevar else self._yf_value

    def _lapack_setup_wrapper(self, jacobian_method, state, prefactor):
        j = jacobian_method(state) * prefactor
        j[self._diag_indices] -= 1.
        self._left_hand_side_inverse_operator = lapack_lu_factor(j)[:2]

    def _superlu_setup_wrapper(self, jacobian_method, state, prefactor):
        j = jacobian_method(state)
        j *= prefactor
        j[self._diag_indices] -= 1.
        j = sparse_csc_matrix(j)
        j.eliminate_zeros()
        self._left_hand_side_inverse_operator = superlu_factor(j)

    def _lapack_solve(self, residual):
        return lapack_lu_solve(self._left_hand_side_inverse_operator[0],
                               self._left_hand_side_inverse_operator[1],
                               residual)[0], 1, True

    def _superlu_solve(self, residual):
        return self._left_hand_side_inverse_operator.solve(residual), 1, True

    def _extra_logger_log(self, state, *args, **kwargs):
        T = state[self._temperature_index]
        T0 = self.initial_temperature
        return f'{T:>10.2f} | {T-T0:>10.2f}|'

    @property
    def initial_state(self):
        """Obtain this reactor's initial state vector"""
        return self._initial_state

    @property
    def current_state(self):
        """Obtain this reactor's final state vector"""
        return self._current_state

    @property
    def initial_temperature(self):
        """Obtain this reactor's initial temperature"""
        return self._initial_temperature

    @property
    def current_temperature(self):
        """Obtain this reactor's current temperature"""
        return self._current_temperature

    @property
    def initial_pressure(self):
        """Obtain this reactor's initial pressure"""
        return self._initial_pressure

    @property
    def current_pressure(self):
        """Obtain this reactor's current pressure"""
        return self._current_pressure

    @property
    def initial_mass_fractions(self):
        """Obtain this reactor's initial mass fractions"""
        return self._initial_mass_fractions

    @property
    def current_mass_fractions(self):
        """Obtain this reactor's current mass fractions"""
        return self._current_mass_fractions

    @property
    def initial_time(self):
        """Obtain this reactor's initial mass fractions"""
        return self._initial_time

    @property
    def current_time(self):
        """Obtain this reactor's current mass fractions"""
        return self._current_time

    @property
    def gas(self):
        """Obtain a cantera gas object for the mechanism in this reactor"""
        return self._gas

    @property
    def n_species(self):
        return self._n_species

    @property
    def n_reactions(self):
        return self._n_reactions

    @classmethod
    def get_supported_reactor_shapes(self):
        """Obtain a list of supported reactor geometries"""
        return HomogeneousReactor._shape_dict.keys()

    def integrate(self,
                  termination,
                  first_time_step=1.e-6,
                  max_time_step=1.e6,
                  minimum_time_step_count=40,
                  transient_tolerance=1.e-10,
                  write_log=False,
                  log_rate=100,
                  maximum_steps_per_jacobian=1,
                  nonlinear_solve_tolerance=1.e-12,
                  linear_solver='lapack',
                  plot=None,
                  stepper_type=ESDIRK64,
                  nlsolver_type=SimpleNewtonSolver,
                  stepcontrol_type=PIController,
                  extra_governor_args=dict(),
                  extra_stepper_args=dict(),
                  extra_nlsolver_args=dict(),
                  extra_stepcontrol_args=dict(),
                  save_frequency=1,
                  save_first_and_last_only=False):
        """Base method for reactor integration

        Parameters
        ----------
        termination : Termination object as in spitfire.time.governor
            how integration is stopped - instead of calling integrate() directly, use the integrate_to_time(), integrate_to_steady(), etc. methods of this class
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The largest time step the time stepper is allowed to take
        minimum_time_step_count : int
            The minimum number of time steps to run (helpful for slowly evolving simulations, for instance those with low starting temperatures)
        transient_tolerance : float
            the target temporal error for transient integration
        write_log : bool
            whether or not to print integration statistics and status during the simulation
        log_rate : int
            how often to print log information
        maximum_steps_per_jacobian : int
            maximum number of steps Spitfire allows before the Jacobian must be re-evaluated - keep low for robustness, try to increase for performance on large mechanisms
        nonlinear_solve_tolerance : float
            tolerance for the nonlinear solver used in implicit time stepping (optional, default: 1e-12)
        linear_solver : str
            which linear solver to use, at the moment either 'lapack' (dense, direct) or 'superlu' (sparse, direct) are available
        plot : list
            List of variables (temperature and/or specific species names) to be plotted after the time integration completes.
            No plot is shown if a list is not provided.
            Temperature is plotted in the first subplot if any list of variables is provided for plotting (even if temperature is not specified in the list of variables).
            Species mass fractions will be plotted in a second subplot if any species names are provided in the list of variables.
        stepper_type : spitfire.time.TimeStepper
            which (single step) stepper method to use (optional, default: ESDIRK64)
        nlsolver_type : spitfire.time.NonlinearSolver
            which nonlinear solver method to use (optional, default: SimpleNewtonSolver)
        stepcontrol_type : spitfire.time.StepControl
            which time step adaptation method to use (optional, default: PIController)
        extra_governor_args : dict
            any extra arguments to specify on the spitfire.time.Governor object that drives time integration
        extra_stepper_args : dict
            extra arguments to specify on the spitfire.time.TimeStepper object
        extra_nlsolver_args : dict
            extra arguments to specify on the spitfire.time.NonlinearSolver object
        extra_stepcontrol_args : dict
            extra arguments to specify on the spitfire.time.StepControl object
        save_frequency : int
            how many steps are taken between solution data and times being saved (default: 1)
        save_first_and_last_only : bool
            whether or not to retain all data (False, default) or only the first and last solutions
        Returns
        -------
            a library containing temperature, mass fractions, and density (isochoric) or pressure (isobaric) over time
        """

        data_holder = SaveAllDataToList(self._current_state,
                                        self._current_time,
                                        save_frequency=save_frequency,
                                        save_first_and_last_only=save_first_and_last_only)

        # build the integration governor and set attributes
        governor = Governor()
        governor.termination_criteria = termination
        governor.minimum_time_step_count = minimum_time_step_count
        governor.projector_setup_rate = maximum_steps_per_jacobian
        governor.do_logging = write_log
        governor.log_rate = log_rate
        governor.extra_logger_log = self._extra_logger_log
        governor.extra_logger_title_line1 = self._extra_logger_title_line1
        governor.extra_logger_title_line2 = self._extra_logger_title_line2
        governor.clip_to_positive = True
        governor.norm_weighting = 1. / self._variable_scales
        governor.custom_post_process_step = data_holder.save_data
        for a in extra_governor_args:
            setattr(governor, a, extra_governor_args[a])

        # build the step controller and set attributes
        step_control_args = {'first_step': first_time_step,
                             'max_step': max_time_step,
                             'target_error': transient_tolerance}
        step_control_args.update(extra_stepcontrol_args)
        step_controller = stepcontrol_type(**step_control_args)

        # build the nonlinear solver and set attributes
        nonlinear_solver_args = {'evaluate_jacobian_every_iter': False,
                                 'norm_weighting': 1. / self._variable_scales,
                                 'tolerance': nonlinear_solve_tolerance}
        nonlinear_solver_args.update(extra_nlsolver_args)
        newton_solver = nlsolver_type(**nonlinear_solver_args)

        # build the stepper method and set attributes
        stepper_args = {'nonlinear_solver': newton_solver, 'norm_weighting': 1. / self._variable_scales}
        stepper_args.update(extra_stepper_args)
        stepper = stepper_type(**stepper_args)

        # build the rhs and projector methods and do the integration
        if self._configuration == 'isobaric':
            def rhs_method(time, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(time)
                self._update_heat_transfer_parameters(time)
                self._griffon.reactor_rhs_isobaric(state, self._initial_pressure,
                                                   self._tf_value, self._yf_value, self._tau_value,
                                                   self._tc_value, self._tr_value,
                                                   self._cc_value, self._re_value,
                                                   self._surface_area_to_volume,
                                                   self._heat_transfer_option, self._is_open, k)
                return k

            def jac_method(state):
                k = np.zeros(self._n_equations)
                j = np.zeros(self._n_equations * self._n_equations)
                self._griffon.reactor_jac_isobaric(state, self._initial_pressure,
                                                   self._tf_value, self._yf_value, self._tau_value,
                                                   self._tc_value, self._tr_value,
                                                   self._cc_value, self._re_value,
                                                   self._surface_area_to_volume,
                                                   self._heat_transfer_option, self._is_open,
                                                   self._rates_sensitivity_option, self._sensitivity_transform_option,
                                                   k, j)
                return j.reshape((self._n_equations, self._n_equations), order='F')

        elif self._configuration == 'isochoric':
            def rhs_method(time, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(time)
                self._update_heat_transfer_parameters(time)
                self._griffon.reactor_rhs_isochoric(state,
                                                    self._rf_value, self._tf_value, self._yf_value, self._tau_value,
                                                    self._tc_value, self._tr_value,
                                                    self._cc_value, self._re_value,
                                                    self._surface_area_to_volume,
                                                    self._heat_transfer_option, self._is_open, k)
                return k

            def jac_method(state):
                k = np.zeros(self._n_equations)
                j = np.zeros(self._n_equations * self._n_equations)
                self._griffon.reactor_jac_isochoric(state,
                                                    self._rf_value, self._tf_value, self._yf_value, self._tau_value,
                                                    self._tc_value, self._tr_value,
                                                    self._cc_value, self._re_value,
                                                    self._surface_area_to_volume,
                                                    self._heat_transfer_option, self._is_open,
                                                    self._rates_sensitivity_option, k, j)
                return j.reshape((self._n_equations, self._n_equations), order='F')

        setup_wrapper = getattr(self, '_' + linear_solver + '_setup_wrapper')
        setup_method = lambda state, prefactor: setup_wrapper(jac_method, state, prefactor)

        solve_method = self._lapack_solve if linear_solver == 'lapack' else self._superlu_solve

        _, current_state, time, _ = governor.integrate(right_hand_side=rhs_method,
                                                       projector_setup=setup_method,
                                                       projector_solve=solve_method,
                                                       initial_condition=self._current_state,
                                                       initial_time=self._current_time,
                                                       controller=step_controller,
                                                       method=stepper)
        self._current_state = np.copy(current_state)
        self._current_time = np.copy(time)

        time_dimension = Dimension('time', data_holder.t_list)
        output_library = Library(time_dimension)

        if self._configuration == 'isobaric':
            self._current_pressure = self._initial_pressure
            self._current_temperature = current_state[0]
            ynm1 = current_state[1:]
            self._current_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
            output_library['temperature'] = data_holder.solution_list[:, 0]
            output_library['pressure'] = self.current_pressure + np.zeros_like(output_library['temperature'])

        elif self._configuration == 'isochoric':
            ynm1 = current_state[2:]
            self._current_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
            self._gas.TDY = current_state[1], current_state[0], self._current_mass_fractions
            self._current_pressure = self._gas.P
            self._current_temperature = current_state[1]
            output_library['density'] = data_holder.solution_list[:, 0]
            output_library['temperature'] = data_holder.solution_list[:, 1]

        species_names = self._mechanism.species_names
        output_library['mass fraction ' + species_names[-1]] = np.ones_like(output_library['temperature'])
        for i, s in enumerate(species_names[:-1]):
            output_library['mass fraction ' + s] = data_holder.solution_list[:, self._temperature_index + 1 + i]
            output_library['mass fraction ' + species_names[-1]] -= output_library['mass fraction ' + s]

        if plot is not None:
            t = output_library.time_values * 1.e3
            T = output_library['temperature']
            if plot == ['temperature']:  # only plot temperature if it is the only requested variable
                plt.semilogx(t, T)
                plt.xlabel('time (ms)')
                plt.ylabel('Temperature (K)')
                plt.grid()
                plt.show()
            else:  # if variables other than temperature are included in the list, plot those in a separate subplot
                f, (axT, axY) = plt.subplots(2, sharex=True, sharey=False)
                axT.semilogx(t, T)  # always plot T
                axT.set_ylabel('Temperature (K)')
                axT.grid()
                for species_vars in plot:
                    if species_vars is not 'temperature':  # separate subplot for species mass fractions
                        Y = output_library['mass fraction ' + species_vars]
                        axY.loglog(t, Y, label=species_vars)
                axY.set_xlabel('time (ms)')
                axY.set_ylabel('Mass Fractions')
                axY.set_ylim([1.e-12, 1.e0])
                axY.grid()
                axY.legend()
                plt.show()

        return output_library

    def integrate_to_steady(self, steady_tolerance=1.e-4, **kwargs):
        """Integrate a reactor until steady state is reached

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        Returns
        -------
            a library containing temperature, mass fractions, and density (isochoric) or pressure (isobaric) over time
        """
        return self.integrate(Steady(steady_tolerance), **kwargs)

    def integrate_to_time(self, final_time, **kwargs):
        """Integrate a reactor until it reaches a specified simulation time

        Parameters
        ----------
        final_time : float
            time at which integration ceases
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        Returns
        -------
            a library containing temperature, mass fractions, and density (isochoric) or pressure (isobaric) over time
        """
        return self.integrate(FinalTime(final_time), **kwargs)

    def integrate_to_steady_after_ignition(self,
                                           steady_tolerance=1.e-4,
                                           delta_temperature_ignition=400.,
                                           **kwargs):
        """Integrate a reactor until steady state is reached after ignition (based on temperature) has occurred.
            This is helpful in slowly-evolving systems whose initial residual may be lower than the prescribed tolerance.

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        Returns
        -------
            a library containing temperature, mass fractions, and density (isochoric) or pressure (isobaric) over time
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError(
                'integrate_to_steady_after_ignition() called on an isothermal reactor! This is not currently supported!')
        else:
            T0 = self._initial_temperature
            Tidx = self._temperature_index

            def stop_at_steady_after_ignition(state, t, nt, residual):
                has_ignited = state[Tidx] > (T0 + delta_temperature_ignition)
                is_steady = residual < steady_tolerance
                return not (has_ignited and is_steady)

            return self.integrate(CustomTermination(stop_at_steady_after_ignition), **kwargs)

    def compute_ignition_delay(self,
                               delta_temperature_ignition=400.,
                               minimum_allowable_residual=1.e-12,
                               return_solution=False,
                               **kwargs):
        """Integrate in time until ignition (exceeding a specified threshold of the increase in temperature)

        Parameters
        ----------
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K
        minimum_allowable_residual : float
            how small the residual can be before the reactor is deemed to 'never' ignite, default is 1.e-12
        return_solution : bool
            whether or not to return the solution trajectory in addition to the ignition delay, as a tuple, (t, library)
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        Returns
        -------
            the ignition delay of the reactor, in seconds, and optionally a library containing temperature, mass fractions, and density (isochoric) or pressure (isobaric) over time
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError(
                'compute_ignition_delay() called on an isothermal reactor! This is not currently supported!')
        else:
            T0 = self._initial_temperature
            Tidx = self._temperature_index

            def stop_at_ignition(state, t, nt, residual):
                has_ignited = state[Tidx] > (T0 + delta_temperature_ignition)
                if residual > minimum_allowable_residual:
                    return not has_ignited
                else:
                    error_msg = f'From compute_ignition_delay(): '
                    f'residual < minimum allowable value ({minimum_allowable_residual}),'
                    f' suggesting that the reactor will not ignite.'
                    f'\nNote that you can set this value with the "minimum_allowable_residual" argument.'
                    f'\nIt is advised that you also pass write_log=True to observe progress of the simulation '
                    f'in case it is running perpetually.'
                raise ValueError(error_msg)

        output_library = self.integrate(CustomTermination(stop_at_ignition), **kwargs)
        tau_ignition = output_library.time_values[-1]
        if return_solution:
            return tau_ignition, output_library
        else:
            return tau_ignition
