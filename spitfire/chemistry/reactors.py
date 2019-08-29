"""
This module contains the HomogeneousReactor class that provides a high-level interface for 0-D reactors
"""

from spitfire.time.governor import Governor, CustomTermination, Steady, FinalTime
from spitfire.time.methods import ESDIRK64
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
import numpy as np
from numpy import zeros, array, hstack, sqrt, sum
from scipy.linalg.lapack import dgetrf as lapack_lu_factor
from scipy.linalg.lapack import dgetrs as lapack_lu_solve
from scipy.sparse.linalg import splu as superlu_factor
from scipy.sparse import csc_matrix as sparse_csc_matrix
from cantera import gas_constant
from scipy.linalg import eig, eigvals
import matplotlib.pyplot as plt


class HomogeneousReactor(object):
    """A class for solving zero-dimensional reactors

    **Constructor**: specify a mechanism, initial mixture, reactor specifications, and thermochemical property evaluation

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
        how the chemical source term Jacobian is formed, either 'exact' or 'no-TBAF' which ignores third-body and falloff sensitivities
    sensitivity_transform_type : str
        how the Jacobian is transformed for isobaric systems, currently only 'exact' is supported
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
                 rates_sensitivity_type='exact',
                 sensitivity_transform_type='exact'):

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

        self._rates_sensitivity_option = {'exact': 0, 'no-TBAF': 1}[rates_sensitivity_type]
        self._sensitivity_transform_option = {'exact': 0}[sensitivity_transform_type]
        self._is_open = self._mass_transfer == 'open'
        self._heat_transfer_option = {'adiabatic': 0, 'isothermal': 1, 'diathermal': 2}[self._heat_transfer]

        self._gas = initial_mixture
        self._initial_pressure = initial_mixture.P
        self._initial_temperature = initial_mixture.T
        self._initial_mass_fractions = initial_mixture.Y
        self._final_pressure = None
        self._final_temperature = None
        self._final_mass_fractions = None

        self._mechanism = mech_spec
        self._griffon = self._mechanism.griffon

        self._n_species = self._gas.n_species
        self._n_reactions = self._gas.n_reactions
        self._n_equations = self._n_species if self._configuration == 'isobaric' else self._n_species + 1

        self._solution_times = []

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

        self._final_state = np.copy(self._initial_state)

        self._variable_scales = np.ones(self._n_equations)
        self._variable_scales[self._temperature_index] = 1.e3

        self._left_hand_side_inverse_operator = None
        self._diag_indices = np.diag_indices(self._n_equations)

        self._V_stoich = self._gas.product_stoich_coeffs() - self._gas.reactant_stoich_coeffs()
        self._cema_Vmod = np.zeros((self._n_equations, self._n_reactions))
        self._cema_Vmod[self._temperature_index + 1:, :] = np.copy(self._V_stoich[:-1, :])
        self._cema_Vmod[self._temperature_index, :] = \
            (self._gas.standard_enthalpies_RT * gas_constant * 298.).T.dot(self._V_stoich)

        self._cema_eigenvalue = False
        self._cema_explosion_indices = False
        self._cema_participation_indices = False
        self._cema_secondary_mode = False

        self._insitu_process_rates = False
        self._insitu_process_mass_fractions = False
        self._insitu_process_mole_fractions = False

        self._enabled_insitu_processors = set()
        self._enabled_cantera_insitu_processors = set()
        self._insitu_processed_data = dict()
        self._no_insitu_processors_enabled = True

        self._ignition_delay = None
        self._delta_temperature_ignition = 400.

        self._extra_logger_title_line1 = '   {:<16}  {:<16}'.format('Reactor', 'Temperature')
        self._extra_logger_title_line2 = '  {:<16}  {:<16}'.format('Temperature (K)', '- Initial (K)')

    def insitu_process_cantera_method(self, label, method=None, index=None):
        """Add a general cantera function as an in situ processor

        Parameters
        ----------
        label : str
            the label of the computed result (also the cantera method to compute if the method parameter is not given)
        method : str
            the cantera method to evaluate
        index : str or int
            the integer index or species name of the vector element of interest, if the cantera function returns a vector
        """
        method = label if method is None else method
        self._enabled_cantera_insitu_processors.add((label, method, index))

    def insitu_process_cema(self,
                            explosion_indices=False,
                            participation_indices=False,
                            secondary_mode=False):
        """Turn on chemical explosive mode analysis, at least computing the explosive eigenvalue

        Parameters
        ----------
        explosion_indices : bool
            whether or not to compute variable explosion indices
        participation_indices : bool
            whether or not to compute reaction participation indices
        secondary_mode : bool
            whether or not to compute the secondary explosive eigenvalue and (if specified already) its associated explosion/participation indices
        """
        self._cema_eigenvalue = True
        self._cema_explosion_indices = explosion_indices
        self._cema_participation_indices = participation_indices
        self._cema_secondary_mode = secondary_mode

        self._enabled_insitu_processors.add('cema-lexp1')
        if secondary_mode:
            self._enabled_insitu_processors.add('cema-lexp2')
        if explosion_indices:
            for species in ['T'] + self._gas.species_names[:-1]:
                self._enabled_insitu_processors.add('cema-ei1 ' + species)
                if secondary_mode:
                    self._enabled_insitu_processors.add('cema-ei2 ' + species)
        if participation_indices:
            for i in range(self._gas.n_reactions):
                self._enabled_insitu_processors.add('cema-pi1 ' + str(i))
                if secondary_mode:
                    self._enabled_insitu_processors.add('cema-pi2 ' + str(i))

    def insitu_process_quantity(self, key):
        """Specify that a quantity be processed in situ.

            Available keys: 'temperature', 'density', 'pressure', 'energy', 'enthalpy', 'heat capacity cv', 'heat capacity cp',
            'mass fractions', 'mole fractions', 'production rates', 'heat release rate', 'eigenvalues'

            If a key is not in this list then cantera will be used.
            This will cause a failure if cantera can not compute the requested quantity.

        Parameters
        ----------
        key : str
            the quantity to process
        """
        if isinstance(key, list):
            for key1 in key:
                self.insitu_process_quantity(key1)
        else:
            if key in ['heat capacity cv', 'heat capacity cp',
                       'pressure', 'density', 'temperature',
                       'enthalpy', 'energy', 'heat release rate', 'eigenvalues']:
                self._enabled_insitu_processors.add(key)
            elif key == 'mass fractions':
                self._insitu_process_mass_fractions = True
                for species in self._gas.species_names:
                    self._enabled_insitu_processors.add('mass fraction ' + species)
            elif key == 'mole fractions':
                self._insitu_process_mole_fractions = True
                for species in self._gas.species_names:
                    self._enabled_insitu_processors.add('mole fraction ' + species)
            elif key == 'production rates':
                self._insitu_process_rates = True
                for species in self._gas.species_names:
                    self._enabled_insitu_processors.add('production rate ' + species)
            else:
                self.insitu_process_cantera_method(key)

    def _do_cema_on_one_state(self, state):
        lexp1, lexp2, ei1, ei2, pi1, pi2 = None, None, None, None, None, None

        # get an appropriate Jacobian
        jac = np.zeros(self._n_equations * self._n_equations)
        k = np.zeros(self._n_equations)
        if self._configuration == 'isobaric':
            p = self._initial_pressure
            if self._heat_transfer == 'adiabatic' or self._heat_transfer == 'diathermal':
                self._griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                                   0, 0, 0, False, 0, 0, k, jac)
                jac = jac.reshape((self._n_equations, self._n_equations), order='F')
            elif self._heat_transfer == 'isothermal':
                self._griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                                   0, 0, 1, False, 0, 0, k, jac)
                jac = jac.reshape((self._n_equations, self._n_equations), order='F')

        elif self._configuration == 'isochoric':
            p = self._griffon.ideal_gas_pressure(state[0], state[1], hstack((state[2:], 1. - sum(state[2:]))))
            if self._heat_transfer == 'adiabatic' or self._heat_transfer == 'diathermal':
                self._griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                                    0, 0, 0, 0, 0, 0, 0, False, 0, k, jac)
                jac = jac.reshape((self._n_equations, self._n_equations), order='F')
            elif self._heat_transfer == 'isothermal':
                self._griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                                    0, 0, 0, 0, 0, 0, 1, False, 0, k, jac)
                jac = jac.reshape((self._n_equations, self._n_equations), order='F')

        # do cema with the Jacobian
        if self._cema_explosion_indices or self._cema_participation_indices:
            w, vl, vr = eig(jac, left=True, right=True)
        else:
            w = eigvals(jac)

        realparts = np.real(w)
        threshold = 1.e-4
        realparts[(realparts > -threshold) & (realparts < threshold)] = -np.inf
        exp_idx1 = np.argmax(realparts)
        real_parts_without_1 = np.delete(realparts, exp_idx1)
        exp_idx2 = np.argmax(real_parts_without_1)
        lexp1 = realparts[exp_idx1]
        lexp2 = realparts[exp_idx2]

        if self._cema_explosion_indices or self._cema_participation_indices:
            exp_rvec1 = vr[:, exp_idx1]
            exp_lvec1 = vl[:, exp_idx1]
            exp_rvec2 = vr[:, exp_idx2]
            exp_lvec2 = vl[:, exp_idx2]
            ep_list1 = np.abs(np.real(exp_lvec1) * np.real(exp_rvec1))
            ei1 = ep_list1 / np.sum(np.abs(ep_list1))
            ep_list2 = np.abs(np.real(exp_lvec2) * np.real(exp_rvec2))
            ei2 = ep_list2 / np.sum(np.abs(ep_list2))

        if self._cema_participation_indices:
            if self._configuration == 'isobaric':
                ynm1 = state[1:]
                self._gas.TPY = state[0], self._initial_pressure, hstack((ynm1, 1. - sum(ynm1)))
            elif self._configuration == 'isochoric':
                ynm1 = state[2:]
                self._gas.TDY = state[1], state[0], hstack((ynm1, 1. - sum(ynm1)))
            qnet = self._gas.net_rates_of_progress
            pi1 = np.abs(exp_lvec1.dot(self._cema_Vmod) * qnet)
            pi1 /= np.sum(np.abs(pi1))
            pi2 = np.abs(exp_lvec2.dot(self._cema_Vmod) * qnet)
            pi2 /= np.sum(np.abs(pi2))

        return lexp1, lexp2, ei1, ei2, pi1, pi2

    def _do_insitu_processing(self, t, state, data_dict=None, *args, **kwargs):
        saving_trajectory_data = data_dict is None or data_dict == self._insitu_processed_data
        if saving_trajectory_data:
            data_dict = self._insitu_processed_data
        if self._no_insitu_processors_enabled and saving_trajectory_data:
            self._solution_times.append(t)
            return
        else:
            if saving_trajectory_data:
                self._solution_times.append(t)

            if self._configuration == 'isobaric':
                if isinstance(state, np.ndarray):
                    p = self._initial_pressure
                    T = state[0]
                    ynm1 = state[1:]
                    y = hstack((ynm1, 1. - sum(ynm1)))
                else:
                    p = self._initial_pressure
                    T = state.T
                    y = state.Y
                    state = np.hstack([T, y[:-1]])
                rho = self._griffon.ideal_gas_density(p, T, y)
            elif self._configuration == 'isochoric':
                if isinstance(state, np.ndarray):
                    rho = state[0]
                    T = state[1]
                    ynm1 = state[2:]
                    y = hstack((ynm1, 1. - sum(ynm1)))
                else:
                    rho = state.density_mass
                    T = state.T
                    y = state.Y
                    state = np.hstack([rho, T, y[:-1]])
                p = self._griffon.ideal_gas_pressure(rho, T, y)

            # check for species production rates and save them if necessary
            if self._insitu_process_rates:
                w = np.zeros(self._n_species)
                self._griffon.production_rates(T, rho, y, w)
                for i in range(self._n_species):
                    data_dict['production rate ' + self._gas.species_names[i]].append(w[i])

            # check for the heat release rate specially, as it needs the production rates
            if 'heat release rate' in self._enabled_insitu_processors:
                if not self._insitu_process_rates:
                    w = np.zeros(self._n_species)
                    self._griffon.production_rates(T, rho, y, w)
                if self._configuration == 'isobaric':
                    h = np.zeros(self._n_species)
                    self._griffon.species_enthalpies(T, h)
                    cp = self._griffon.cp_mix(T, y)
                    data_dict['heat release rate'].append(-np.sum(w * h) / rho / cp)
                elif self._configuration == 'isochoric':
                    e = np.zeros(self._n_species)
                    self._griffon.species_energies(T, e)
                    cv = self._griffon.cv_mix(T, y)
                    data_dict['heat release rate'].append(-np.sum(w * e) / rho / cv)

            # check for mass and mole fractions and save them all at once
            if self._insitu_process_mass_fractions:
                for i in range(self._n_species):
                    data_dict['mass fraction ' + self._gas.species_names[i]].append(y[i])
            if self._insitu_process_mole_fractions:
                x = np.zeros(self._n_species)
                self._griffon.mole_fractions(y, x)
                for i in range(self._n_species):
                    data_dict['mole fraction ' + self._gas.species_names[i]].append(x[i])

            # handle remaining non-general-cantera processors
            for pp in self._enabled_insitu_processors:
                if pp == 'pressure':
                    data_dict[pp].append(p)
                elif pp == 'density':
                    data_dict[pp].append(rho)
                elif pp == 'temperature':
                    data_dict[pp].append(T)
                elif pp == 'eigenvalues':
                    # get an appropriate Jacobian
                    jac = np.zeros(self._n_equations * self._n_equations)
                    k = np.zeros(self._n_equations)
                    if self._configuration == 'isobaric':
                        if self._heat_transfer == 'adiabatic' or self._heat_transfer == 'diathermal':
                            self._griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                                               0, 0, 0, False, 0, 0, k, jac)
                            jac = jac.reshape((self._n_equations, self._n_equations), order='F')
                        elif self._heat_transfer == 'isothermal':
                            self._griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                                               0, 0, 1, False, 0, 0, k, jac)
                            jac = jac.reshape((self._n_equations, self._n_equations), order='F')

                    elif self._configuration == 'isochoric':
                        if self._heat_transfer == 'adiabatic' or self._heat_transfer == 'diathermal':
                            self._griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                                                0, 0, 0, 0, 0, 0, 0, False, 0, k, jac)
                            jac = jac.reshape((self._n_equations, self._n_equations), order='F')
                        elif self._heat_transfer == 'isothermal':
                            self._griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                                                0, 0, 0, 0, 0, 0, 1, False, 0, k, jac)
                            jac = jac.reshape((self._n_equations, self._n_equations), order='F')
                    data_dict[pp].append(eigvals(jac))
                elif pp == 'heat capacity cv':
                    data_dict[pp].append(self._griffon.cv_mix(T, y))
                elif pp == 'heat capacity cp':
                    data_dict[pp].append(self._griffon.cp_mix(T, y))
                elif pp == 'enthalpy':
                    data_dict[pp].append(self._griffon.enthalpy_mix(T, y))
                elif pp == 'energy':
                    data_dict[pp].append(self._griffon.energy_mix(T, y))

            # handle general cantera processors
            if len(self._enabled_cantera_insitu_processors):
                self._gas.TPY = T, p, y
                for label, method, index in self._enabled_cantera_insitu_processors:
                    if index is None:
                        data_dict[label].append(getattr(self._gas, method))
                    else:
                        if isinstance(index, str):
                            data_dict[label].append(
                                getattr(self._gas, method)[self._gas.species_index(index)])
                        else:
                            data_dict[label].append(getattr(self._gas, method)[index])

            # handle chemical explosive mode analysis
            if self._cema_eigenvalue:
                lexp1, lexp2, ei1, ei2, pi1, pi2 = self._do_cema_on_one_state(state)
                data_dict['cema-lexp1'].append(lexp1)
                if self._cema_explosion_indices:
                    for idx, name in enumerate(['T'] + self._gas.species_names[:-1]):
                        data_dict['cema-ei1 ' + name].append(np.copy(ei1[idx]))
                if self._cema_participation_indices:
                    for rxn_index in range(self._gas.n_reactions):
                        data_dict['cema-pi1 ' + str(rxn_index)].append(np.copy(pi1[rxn_index]))

                if self._cema_secondary_mode:
                    data_dict['cema-lexp2'].append(lexp2)
                    if self._cema_explosion_indices:
                        for idx, name in enumerate(['T'] + self._gas.species_names[:-1]):
                            data_dict['cema-ei2 ' + name].append(np.copy(ei2[idx]))
                    if self._cema_participation_indices:
                        for rxn_index in range(self._gas.n_reactions):
                            data_dict['cema-pi2 ' + str(rxn_index)].append(np.copy(pi2[rxn_index]))

    def _initialize_insitu_processing(self):
        if not len(self._enabled_insitu_processors) and not len(self._enabled_cantera_insitu_processors):
            self._no_insitu_processors_enabled = True
            self._do_insitu_processing(0., self._initial_state)
        else:
            self._no_insitu_processors_enabled = False
            for pp in self._enabled_insitu_processors:
                self._insitu_processed_data[pp] = []
            for label, method, index in self._enabled_cantera_insitu_processors:
                self._insitu_processed_data[label] = []

            self._do_insitu_processing(0., self._initial_state)

    def process_quantities_on_state(self, state):
        """Compute the specified in situ quantities on a given reactor state.
        This will return a dictionary that maps each quantity's name to its computed value for this state.

        Parameters
        ----------
        state : np.ndarray, Cantera.Quantity (a Spitfire stream), or Cantera.gas object
            the thermochemical state at which quantities will be computed
        """
        self._no_insitu_processors_enabled = False
        data_dict = dict()
        for pp in self._enabled_insitu_processors:
            data_dict[pp] = []
        for label, method, index in self._enabled_cantera_insitu_processors:
            data_dict[label] = []
        self._do_insitu_processing(0., state, data_dict)
        for key in data_dict:
            data_dict[key] = array(data_dict[key])
        return data_dict

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

    def _stop_at_ignition(self, state, t, nt, residual):
        has_ignited = (state[self._temperature_index] > (self._initial_temperature + self._delta_temperature_ignition))
        is_not_steady = residual > self._minimum_allowable_residual_for_ignition
        if is_not_steady:
            return not has_ignited
        else:
            return False

    def _stop_at_steady_after_ignition(self, state, t, nt, residual):
        has_ignited = (state[self._temperature_index] > (self._initial_temperature + self._delta_temperature_ignition))
        is_steady = residual < self.steady_tolerance
        return not (has_ignited and is_steady)

    def _extra_logger_log(self, state, *args, **kwargs):
        return '{:>16.3f}  {:>16.3f}'.format(state[self._temperature_index],
                                             state[self._temperature_index] - self.initial_temperature)

    @property
    def initial_state(self):
        """Obtain this reactor's initial state vector"""
        return self._initial_state

    @property
    def final_state(self):
        """Obtain this reactor's final state vector"""
        return self._final_state

    @property
    def initial_temperature(self):
        """Obtain this reactor's initial temperature"""
        return self._initial_temperature

    @property
    def initial_pressure(self):
        """Obtain this reactor's initial pressure"""
        return self._initial_pressure

    @property
    def initial_mass_fractions(self):
        """Obtain this reactor's initial mass fractions"""
        return self._initial_mass_fractions

    @property
    def initial_explosive_eigenvalue(self):
        return self._do_cema_on_one_state(self._initial_state)[0]

    @property
    def gas(self):
        return self._gas

    @property
    def n_species(self):
        return self._n_species

    @property
    def n_reactions(self):
        return self._n_reactions

    @property
    def final_temperature(self):
        """Obtain this reactor's final temperature"""
        return self._final_temperature

    @property
    def final_pressure(self):
        """Obtain this reactor's final pressure"""
        return self._final_pressure

    @property
    def final_mass_fractions(self):
        """Obtain this reactor's final mass fractions"""
        return self._final_mass_fractions

    @property
    def solution_times(self):
        """Obtain this reactor's integration times"""
        return array(self._solution_times)

    def trajectory_data(self, key):
        """Obtain the simulation data associated with a particular key/label (processed in situ)"""
        if key not in self._insitu_processed_data.keys():
            print('Available data:', self._insitu_processed_data.keys())
            raise ValueError('data identifier ' + str(key) + ' is not valid!')
        else:
            return array(self._insitu_processed_data[key])

    def ignition_delay(self, delta_temperature_ignition=None):
        """Find the time at which ignition occurred in a pre-run simulation (based on temperature increase)

        Parameters
        ----------
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K

        Returns
        -------
            the ignition delay of the pre-simulated reactor, in seconds
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError('ignition_delay() called on an isothermal reactor! This is not currently supported!')
        else:
            if delta_temperature_ignition is not None:
                self._delta_temperature_ignition = delta_temperature_ignition
            T = self.trajectory_data('temperature')
            dT = T - T[0]
            return self.solution_times[np.argmin(np.abs(dT - self._delta_temperature_ignition))]

    @classmethod
    def get_supported_reactor_shapes(self):
        """Obtain a list of supported reactor geometries"""
        return HomogeneousReactor._shape_dict.keys()

    def temporary_alc_in_tau(self, taus, temps):
        # import matplotlib.pyplot as plt

        if self._configuration == 'isobaric':
            def rhs_method(t, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(t)
                self._update_heat_transfer_parameters(t)
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
            def rhs_method(t, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(t)
                self._update_heat_transfer_parameters(t)
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

        tol = 1.e-8
        dh = 1.e-4
        h = 0.

        q = np.copy(self._initial_state)
        p = np.copy(self._tau_value)

        p_target = 1.e-6
        a_sign = -1.

        # plt.ion()
        # fig, axarray = plt.subplots(2, 1)
        # for tau, temp in zip(taus, temps):
        #     axarray[1].semilogx(tau, temp, 'sc', markersize=10, markeredgecolor='k')

        total_iter = 0
        c = 0
        has_switched = False
        p_list = [np.copy(p)]
        T_list = [np.copy(q[0])]
        h_list = [np.copy(dh)]
        # hp_line, = axarray[0].loglog(h_list, p_list, 'k-')
        # pT_line, = axarray[1].semilogx(p_list, T_list, 'k-')
        # hp_pt, = axarray[0].loglog(h_list, p_list, 'k*')
        # pT_pt, = axarray[1].semilogx(p_list, T_list, 'k*')
        while p > p_target:
            c += 1

            # a single local nonlinear problem
            # save (q0, p0) and set up the linear system solvers

            def H_fun(q, p):
                self._tau_value = p
                return rhs_method(0., q)

            neq = self._n_equations

            q0 = np.copy(q)
            p0 = np.copy(p)

            # setting up Newton's method on this local problem requires the tangent vectors at (q0,p0)
            # set up the Hq linear solves
            J = jac_method(q)
            J_lu = lapack_lu_factor(J)[:2]

            q = np.copy(q0)
            p = np.copy(p0)

            # finite difference the parameter Jacobian Hp for now
            Hp = (H_fun(q0, p0 + 1.e-8) - H_fun(q0, p0)) / 1.e-8

            # compute the tangent vectors
            phi = np.zeros_like(Hp)
            v = np.zeros_like(Hp)
            w = np.zeros_like(Hp)
            phi = lapack_lu_solve(J_lu[0], J_lu[1], -Hp)[0]
            a = a_sign / np.sqrt(1. + phi.dot(phi))
            qdot0 = a * phi
            pdot0 = np.copy(a)
            # compute the auxiliary equation Jacobian
            Nq = qdot0.T
            Np = pdot0

            # enter into Newton's method
            for i in range(20):
                # evaluate the residual
                H = H_fun(q, p)
                N = qdot0.dot(q - q0) + pdot0 * (p - p0) - dh

                res_vec = np.hstack((H, N))
                res = res_vec.dot(res_vec)
                if res < tol:
                    break
                else:
                    Hq = jac_method(q)
                    A = np.zeros((neq + 1, neq + 1))
                    A[:-1, :-1] = Hq
                    A[:-1, -1] = Hp
                    A[-1, :-1] = Nq
                    A[-1, -1] = Np
                    b = np.zeros(neq + 1)
                    b[:-1] = H
                    b[-1] = N
                    x = np.linalg.solve(A, -b)
                    dq = x[:-1]
                    dp = x[-1]

                    # do the bordering algorithm to solve the composite linear system
                    # J = getattr(self, '_' + self._heat_transfer + '_jac')(q0)
                    # py_btddod_full_factorize(J, nzi, neq, btl, btp, btf)
                    # Hp = (H_fun(q0, p0 + 1.e-2) - H_fun(q0, p0)) / 1.e-2
                    #
                    # py_btddod_full_solve(J, btl, btp, btf, Hp, nzi, neq, v)
                    # py_btddod_full_solve(J, btl, btp, btf, H, nzi, neq, w)
                    #
                    # dp = -(N - Nq.dot(w)) / (Np - Nq.dot(v))
                    # dq = w + dp * v

                    q += dq
                    p += dp
            h += dh
            dqn = (q - q0) / self._variable_scales
            dtarget = 1.e-10
            dhmax = 1.
            dh = np.min([dhmax, dh * np.sqrt(dtarget / dqn.dot(dqn))])

            p_list.append(np.copy(p))
            h_list.append(np.copy(h))
            T_list.append(np.copy(q[0]))

            if total_iter > 100:
                if pmp0_old * (p - p0) < 0.:
                    a_sign *= -1.
                    has_switched = True
                    print('switch!')
                    # dtarget *= 1.e-4

            print(q[0], p, p - p0, dh)
            pmp0_old = p - p0
            if c == 10000:
                c = 0
                # axarray[0].plot(self._z[1:-1], q[::self._n_equations])
                # axarray[0].loglog(h, p, '.')
                # axarray[1].semilogx(p, q[0], '.')
                # hp_line.set_data(h_list, p_list)
                # axarray[0].set_xlim([0.9 * np.min(h_list[1:]), 1.1 * np.max(h_list)])
                # axarray[0].set_ylim([0.9 * np.min(p_list), 1.1 * np.max(p_list)])
                # hp_pt.set_data(h, p)
                # pT_pt.set_data(p, q[0])
                # pT_line.set_data(p_list, T_list)
                plt.pause(1.e-3)
            total_iter += 1
        # plt.ioff()
        # plt.show()

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
                  extra_stepcontrol_args=dict()):
        """Base method for reactor integration

        Parameters
        ----------
        termination : Termination object as in spitfire.time.governor
            how integration is stopped - instead of calling integrate() directly, use the integrate_to_time(), integrate_to_steady(), etc. methods of this class
        first_time_step : float
            The time step size initially used by the time integrator
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
        linear_solver : str
            which linear solver to use, at the moment either 'lapack' (dense, direct) or 'superlu' (sparse, direct) are available
        plot : list
            List of variables (temperature and/or specific species names) to be plotted after the time integration completes.
            No plot is shown if a list is not provided.
            Temperature is plotted in the first subplot if any list of variables is provided for plotting (even if temperature is not specified in the list of variables).
            Species mass fractions will be plotted in a second subplot if any species names are provided in the list of variables.
        """

        self.insitu_process_quantity(['temperature'])
        if plot is not None:
            if plot != ['temperature']:
                self.insitu_process_quantity(['mass fractions'])

        self._initialize_insitu_processing()

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
        governor.custom_post_process_step = self._do_insitu_processing
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
            def rhs_method(t, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(t)
                self._update_heat_transfer_parameters(t)
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
            def rhs_method(t, state):
                k = np.zeros(self._n_equations)
                self._update_mass_transfer_parameters(t)
                self._update_heat_transfer_parameters(t)
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

        final_state = governor.integrate(right_hand_side=rhs_method,
                                         projector_setup=setup_method,
                                         projector_solve=solve_method,
                                         initial_condition=self._initial_state,
                                         controller=step_controller,
                                         method=stepper)[1]
        self._final_state = np.copy(final_state)

        if self._configuration == 'isobaric':
            self._final_pressure = self._initial_pressure
            self._final_temperature = final_state[0]
            ynm1 = final_state[1:]
            self._final_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
        elif self._configuration == 'isochoric':
            ynm1 = final_state[2:]
            self._final_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
            self._gas.TDY = final_state[1], final_state[0], self._final_mass_fractions
            self._final_pressure = self._gas.P
            self._final_temperature = final_state[1]

        if plot is not None:
            t = self.solution_times * 1.e3
            T = self.trajectory_data('temperature')
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
                        Y = self.trajectory_data('mass fraction ' + species_vars)
                        axY.loglog(t, Y, label=species_vars)
                axY.set_xlabel('time (ms)')
                axY.set_ylabel('Mass Fractions')
                axY.set_ylim([1.e-12, 1.e0])
                axY.grid()
                axY.legend()
                plt.show()

    def integrate_to_steady(self, steady_tolerance=1.e-4, **kwargs):
        """Integrate a reactor until steady state is reached

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined

        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """
        self.steady_tolerance = steady_tolerance
        self.integrate(Steady(steady_tolerance), **kwargs)

    def integrate_to_steady_direct_griffon(self, steady_tolerance=1.e-4, **kwargs):
        final_state = np.zeros(self._n_equations)
        transient_tolerance = 1.e-10 if 'transient_tolerance' not in kwargs else kwargs['transient_tolerance']
        nonlinear_tolerance = 1.e-8
        inv_dof_scales = 1. / self._variable_scales
        if self._configuration == 'isobaric':
            self._griffon.isobaric_esdirk64_solver(self._initial_state, self._initial_pressure,
                                                   self._tf_value, self._yf_value, self._tau_value,
                                                   self._tc_value, self._tr_value,
                                                   self._cc_value, self._re_value,
                                                   self._surface_area_to_volume,
                                                   self._heat_transfer_option, self._is_open,
                                                   self._rates_sensitivity_option,
                                                   self._sensitivity_transform_option,
                                                   transient_tolerance, steady_tolerance,
                                                   nonlinear_tolerance,
                                                   inv_dof_scales,
                                                   40, 10000, 1.e-6, 1.e6, -1.,
                                                   400., 1, final_state)

        if self._configuration == 'isobaric':
            self._final_pressure = self._initial_pressure
            self._final_temperature = final_state[0]
            ynm1 = final_state[1:]
            self._final_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
        elif self._configuration == 'isochoric':
            ynm1 = final_state[2:]
            self._final_mass_fractions = hstack((ynm1, 1. - sum(ynm1)))
            self._gas.TDY = final_state[1], final_state[0], self._final_mass_fractions
            self._final_pressure = self._gas.P
            self._final_temperature = final_state[1]
        self._final_state = np.copy(final_state)

    def integrate_to_time(self, final_time, **kwargs):
        """Integrate a reactor until it reaches a specified simulation time

        Parameters
        ----------
        final_time : float
            time at which integration ceases

        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """
        self.final_time = final_time
        self.integrate(FinalTime(final_time), **kwargs)

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
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError(
                'integrate_to_steady_after_ignition() called on an isothermal reactor! This is not currently supported!')
        else:
            self._delta_temperature_ignition = delta_temperature_ignition
            self.steady_tolerance = steady_tolerance
            self.integrate(CustomTermination(self._stop_at_steady_after_ignition), **kwargs)

    def compute_ignition_delay(self,
                               delta_temperature_ignition=None,
                               minimum_allowable_residual=1.e-12,
                               **kwargs):
        """Integrate in time until ignition (exceeding a specified threshold of the increase in temperature)

        Parameters
        ----------
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K

        minimum_allowable_residual : float
            how small the residual can be before the reactor is deemed to 'never' ignite, default is 1.e-12

        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation

        Returns
        -------
            the ignition delay of the reactor, in seconds
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError(
                'compute_ignition_delay() called on an isothermal reactor! This is not currently supported!')
        else:
            if delta_temperature_ignition is not None:
                self._delta_temperature_ignition = delta_temperature_ignition
            self._minimum_allowable_residual_for_ignition = minimum_allowable_residual
            self._no_insitu_processors_enabled = True
            self.integrate(CustomTermination(self._stop_at_ignition), **kwargs)
            if self._final_temperature < self._initial_temperature + self._delta_temperature_ignition:
                return np.Inf
            else:
                return self._solution_times[-1]

    def compute_ignition_delay_direct_griffon(self,
                                              delta_temperature_ignition=400.,
                                              minimum_allowable_residual=1.e-12,
                                              **kwargs):
        """Integrate in time until ignition (exceeding a specified threshold of the increase in temperature)

        Parameters
        ----------
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K

        minimum_allowable_residual : float
            how small the residual can be before the reactor is deemed to 'never' ignite, default is 1.e-12

        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation

        Returns
        -------
            the ignition delay of the reactor, in seconds
        """
        if self._heat_transfer == 'isothermal':
            raise ValueError(
                'compute_ignition_delay_direct_griffon() called on an isothermal reactor! This is not currently supported!')
        else:
            if delta_temperature_ignition is not None:
                self._delta_temperature_ignition = delta_temperature_ignition
            self._minimum_allowable_residual_for_ignition = minimum_allowable_residual
            self._no_insitu_processors_enabled = True

            transient_tolerance = 1.e-10 if 'transient_tolerance' not in kwargs else kwargs['transient_tolerance']
            steady_tolerance = 1.e-8
            nonlinear_tolerance = 1.e-12

            inv_dof_scales = 1. / self._variable_scales

            if self._configuration == 'isobaric':
                final_state = np.zeros(self._n_equations)
                tau_ign = self._griffon.isobaric_esdirk64_solver(self._initial_state, self._initial_pressure,
                                                                 self._tf_value, self._yf_value, self._tau_value,
                                                                 self._tc_value, self._tr_value,
                                                                 self._cc_value, self._re_value,
                                                                 self._surface_area_to_volume,
                                                                 self._heat_transfer_option, self._is_open,
                                                                 self._rates_sensitivity_option,
                                                                 self._sensitivity_transform_option,
                                                                 transient_tolerance, steady_tolerance,
                                                                 nonlinear_tolerance,
                                                                 inv_dof_scales,
                                                                 40, 10000, 1.e-9, 1.e6, -1.,
                                                                 delta_temperature_ignition, 0, final_state)
                if tau_ign > 0:
                    return tau_ign
                else:
                    return np.Inf
            else:
                raise ValueError(
                    'compute_ignition_delay_direct_griffon() called on an isochoric reactor! This is not currently supported!')
