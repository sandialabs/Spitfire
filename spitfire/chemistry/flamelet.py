"""
This module contains the Flamelet class that provides a high-level interface for nonpremixed flamelets
"""

"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.time.governor import Governor, Steady, FinalTime, CustomTermination
from spitfire.time.methods import ESDIRK64
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
import numpy as np
from numpy import array, hstack, sum
from numpy import any, logical_or, isinf, isnan
from scipy.special import erfinv
from scipy.sparse import csc_matrix
from time import perf_counter
from scipy.sparse.linalg import splu as superlu_factor
from cantera import gas_constant
from numpy.linalg import norm
from scipy.linalg import eig, eigvals
from scipy.interpolate import interp1d
from spitfire.griffon.griffon import py_btddod_full_factorize, py_btddod_full_solve, \
    py_btddod_scale_and_add_diagonal


class Flamelet(object):
    """A class for solving one-dimensional non-premixed flamelets

    **Constructor**: specify boundary streams, mixing rates, etc.

    Parameters
    ----------
    mech_spec : spitfire.chemistry.mechanism.ChemicalMechanismSpec instance
        the mechanism
    initial_condition : str or np.ndarray
        the initial state of the flamelet, either 'equilibrium', 'unreacted', 'linear-TY', or a state vector from another flamelet
    pressure : float
        the pressure of the flamelet
    oxy_stream : Cantera.Quantity (a Spitfire stream) or Cantera.gas object
        the oxidizer stream
    fuel_stream : Cantera.Quantity (a Spitfire stream) or Cantera.gas object
        the fuel stream
    max_dissipation_rate : float
        the maximum dissipation rate
    dissipation_rate : np.ndarray
        the dissipation rate over mixture fraction
    dissipation_rate_form : str
        the form of dissipation rate to use if the maximum value is specified ('Peters' (default) or 'constant')
    lewis_numbers : np.ndarray
        the Lewis numbers - do not use at the moment, as the nonunity Le formulation does not currently have consistent enthalpy fluxes
    grid : np.ndarray
        the location of the grid points (specifying the grid directly invalidates other grid arguments)
    grid_points : int
        the number of grid points to use if Spitfire is to build the grid (if the grid argument is not specified, this is required)
    grid_type : str
        the type of grid, either 'clustered' (default) or 'uniform'
    grid_cluster_intensity : float
        how tightly clustered grid points will be around the grid_cluster_point if grid_type is 'clustered' (default: 4)
    grid_cluster_point : float
        the location of grid point clustering (default is the stoichiometric mixture fraction)
    heat_transfer : str
        whether or not the flamelet is 'adiabatic' or 'nonadiabatic'
    convection_temperature : float
        the convective heat loss reference temperature
    radiation_temperature : float
        the radiation heat loss reference temperature
    convection_coefficient : float
        the convective heat loss coefficient
    radiative_emissivity : float
        the radiative heat loss coefficient
    use_scaled_heat_loss : bool
        whether or not to use a special form of reference temperature and coefficients for heat loss
    include_enthalpy_flux : bool
        whether or not to use a consistent formulation of the enthalpy flux (True) or the simplest flamelet formulation (False)
    include_variable_cp : bool
        whether or not to include variation of the heat capacity (True) or use the simplest flamelet formulation (False)
    rates_sensitivity_type : str
        how the chemical source term Jacobian is formed, either 'exact' or 'no-TBAF' which ignores third-body and falloff sensitivities
    sensitivity_transform_type : str
        how the Jacobian is transformed, currently only 'exact' is supported
    """

    _heat_transfers = ['adiabatic', 'nonadiabatic']
    _initializations = ['unreacted', 'equilibrium', 'Burke-Schumann', 'linear-TY']
    _grid_types = ['uniform', 'clustered']
    _rates_sensitivity_option_dict = {'exact': 0, 'no-TBAF': 1}
    _sensitivity_transform_option_dict = {'exact': 0}

    @classmethod
    def _uniform_grid(cls, grid_points):
        """Make a uniform grid in mixture fraction space with a particular number of grid points

            Parameters
            ----------
            grid_points : int
                the number of uniformly-distributed grid points

            Returns
            -------
            z : array_like
                Locations of grid points, including the boundary points
            dz : array_like
                Spacings between grid points
            """
        z = np.linspace(0., 1., grid_points)
        dz = z[1:] - z[:-1]
        return z, dz

    @classmethod
    def _clustered_grid(cls, grid_points, grid_cluster_point, grid_cluster_intensity=6.):
        """Make a grid in mixture fraction space with clustering around a particular mixture fraction.

            Parameters
            ----------
            grid_points : int
                the number of grid points
            grid_cluster_point : float
                the location around which grid points will be clustered
            grid_cluster_intensity : float, optional
                clustering coefficient, increase for more dense clustering, must be positive, defaults to 6.0

            Returns
            -------
            z : array_like
                Locations of grid points, including the boundary points
            dz : array_like
                Spacings between grid points

            Notes
            -----
            This function uses the clustering method given in [1]_.

            .. [1] J. D. Anderson, "Computational Fluid Dynamics: the Basics with Applications," McGraw-Hill Inc., 1995 pp. 585-588, 1996.
            """

        if grid_cluster_intensity < 1.e-16:
            raise ValueError('cluster_coeff must be strictly positive! Given value: ' + str(grid_cluster_intensity))

        if grid_cluster_point < 0. or grid_cluster_point > 1.:
            raise ValueError('z_cluster must be between 0 and 1! Given value: ' + str(grid_cluster_point))

        z = np.linspace(0., 1., grid_points)
        zo = 1.0 / (2.0 * grid_cluster_intensity) * np.log(
            (1. + (np.exp(grid_cluster_intensity) - 1.) * grid_cluster_point) / (
                    1. + (np.exp(-grid_cluster_intensity) - 1.) * grid_cluster_point))
        a = np.sinh(grid_cluster_intensity * zo)
        for i in range(grid_points):
            z[i] = grid_cluster_point / a * (np.sinh(grid_cluster_intensity * (z[i] - zo)) + a)
        z[-1] = 1.
        dz = z[1:] - z[:-1]
        return z, dz

    @classmethod
    def _compute_dissipation_rate(cls,
                                  mixture_fraction,
                                  max_dissipation_rate,
                                  form='Peters'):
        """Compute the scalar dissipation rate across mixture fraction

            Parameters
            ----------
            mixture_fraction : array_like
                the locations of grid points in mixture fraction space
            max_dissipation_rate : float
                the maximum value of the dissipation rate
            form : str, optional
                the form of the dissipation rate's dependency on mixture fraction, defaults to 'Peters', which
                uses the form of N. Peters, Turbulent Combustion, 2000.
                Specifying anything else will yield a constant scalar dissipation rate.

            Returns
            -------
            x : array_like
                the scalar dissipation rate on the given mixture fraction grid
            """
        if form == 'Peters' or form == 'peters':
            x = max_dissipation_rate * np.exp(-2. * (erfinv(2. * mixture_fraction - 1.)) ** 2)
        else:
            x = np.zeros_like(mixture_fraction)
            x[:] = max_dissipation_rate
        return x

    def _set_heat_transfer_arg_as_np_array(self, input_value, input_name, attr_name):
        if input_value is None:
            raise ValueError('Flamelet specifications: Nonadiabatic heat transfer was selected but no ' + \
                             input_name + ' argument was given.')
        else:
            if isinstance(input_value, float):
                setattr(self, attr_name, input_value + np.zeros(self._nz_interior))
            elif isinstance(input_value, np.ndarray):
                setattr(self, attr_name, input_value)
            else:
                raise ValueError(input_name + ' was not given as a float (constant) or numpy array')

    def __init__(self,
                 mech_spec,
                 initial_condition,
                 pressure,
                 oxy_stream,
                 fuel_stream,
                 max_dissipation_rate=None,
                 stoich_dissipation_rate=None,
                 dissipation_rate=None,
                 dissipation_rate_form='Peters',
                 lewis_numbers=None,
                 grid=None,
                 grid_points=None,
                 grid_type='clustered',
                 grid_cluster_intensity=4.,
                 grid_cluster_point='stoichiometric',
                 heat_transfer='adiabatic',
                 convection_temperature=None,
                 radiation_temperature=None,
                 convection_coefficient=None,
                 radiative_emissivity=None,
                 rates_sensitivity_type='exact',
                 sensitivity_transform_type='exact',
                 include_enthalpy_flux=False,
                 include_variable_cp=False,
                 use_scaled_heat_loss=False):

        self._constructor_arguments = locals()
        del self._constructor_arguments['self']

        # process the mechanism
        self._gas = mech_spec.copy_stream(oxy_stream)
        self._oxy_stream = oxy_stream
        self._fuel_stream = fuel_stream
        self._pressure = pressure
        self._mechanism = mech_spec
        self._n_species = self._gas.n_species
        self._n_reactions = self._gas.n_reactions
        self._n_equations = self._n_species
        self._state_fuel = np.hstack([fuel_stream.T, fuel_stream.Y[:-1]])
        self._state_oxy = np.hstack([oxy_stream.T, oxy_stream.Y[:-1]])

        # build the grid
        # if the grid argument is given, then use that and warn if any other grid arguments ar given
        # if the grid argument is not given, then grid_points must be given, while other grid arguments are all optional
        if grid is not None:
            self._z = np.copy(grid)
            self._dz = self._z[1:] - self._z[:-1]

            warning_message = lambda arg: 'Flamelet specifications: Warning! Setting the grid argument ' \
                                          'nullifies the ' + arg + ' argument.'
            if grid_points is not None:
                print(warning_message('grid_points'))
            if grid_type != 'clustered':
                print(warning_message('grid_type'))
            if grid_cluster_intensity != 4.:
                print(warning_message('grid_cluster_intensity'))
            if grid_cluster_point != 'stoichiometric':
                print(warning_message('grid_cluster_point'))
        else:
            if grid_points is None:
                raise ValueError('Flamelet specifications: one of either grid or grid_points must be given.')
            if grid_type == 'uniform':
                self._z, self._dz = self._uniform_grid(grid_points)
            elif grid_type == 'clustered':
                if grid_cluster_point == 'stoichiometric':
                    grid_cluster_point = self._mechanism.stoich_mixture_fraction(self._fuel_stream, self._oxy_stream)
                self._z, self._dz = self._clustered_grid(grid_points, grid_cluster_point, grid_cluster_intensity)
            else:
                error_message = 'Flamelet specifications: Bad grid_type argument detected: ' + grid_type + '\n' + \
                                '                         Acceptable values: ' + self._grid_types
                raise ValueError(error_message)

        self._nz_interior = self._z.size - 2
        self._n_dof = self._n_equations * self._nz_interior

        # set up the heat transfer
        if heat_transfer not in self._heat_transfers:
            error_message = 'Flamelet specifications: Bad heat_transfer argument detected: ' + heat_transfer + '\n' + \
                            '                         Acceptable values: ' + str(self._heat_transfers)
            raise ValueError(error_message)
        else:
            self._heat_transfer = heat_transfer

        self._T_conv = None
        self._T_rad = None
        self._h_conv = None
        self._h_rad = None
        if self._heat_transfer == 'adiabatic' or (self._heat_transfer == 'nonadiabatic' and use_scaled_heat_loss):
            self._convection_temperature = None
            self._radiation_temperature = None

            warning_message = lambda arg: 'Flamelet specifications: Warning! Setting heat_transfer to adiabatic ' \
                                          'nullifies the ' + arg + ' argument.'
            if convection_temperature is not None:
                print(warning_message('convection_temperature'))
            if radiation_temperature is not None:
                print(warning_message('radiation_temperature'))

            if self._heat_transfer == 'adiabatic':
                self._convection_coefficient = None
                self._radiative_emissivity = None
                if convection_coefficient is not None:
                    print(warning_message('convection_coefficient'))
                if radiative_emissivity is not None:
                    print(warning_message('radiative_emissivity'))
            else:
                self._set_heat_transfer_arg_as_np_array(convection_coefficient, 'convection_coefficient', '_h_conv')
                self._set_heat_transfer_arg_as_np_array(radiative_emissivity, 'radiative_emissivity', '_h_rad')
        else:
            self._set_heat_transfer_arg_as_np_array(convection_temperature, 'convection_temperature', '_T_conv')
            self._set_heat_transfer_arg_as_np_array(radiation_temperature, 'radiation_temperature', '_T_rad')
            self._set_heat_transfer_arg_as_np_array(convection_coefficient, 'convection_coefficient', '_h_conv')
            self._set_heat_transfer_arg_as_np_array(radiative_emissivity, 'radiative_emissivity', '_h_rad')

        # set up the dissipation rate
        if dissipation_rate is not None:
            warning_message = lambda arg: 'Flamelet specifications: Warning! Setting the dissipation_rate argument ' \
                                          'nullifies the ' + arg + ' argument.'
            if max_dissipation_rate is not None:
                warning_message('max_dissipation_rate')

            if stoich_dissipation_rate is not None:
                warning_message('stoich_dissipation_rate')

            if dissipation_rate_form is not None:
                warning_message('dissipation_rate_form')

            self._x = np.copy(dissipation_rate)
            self._max_dissipation_rate = np.max(self._x)
            self._dissipation_rate_form = 'custom'
        else:
            warning_message = lambda arg: 'Flamelet specifications: Warning! Setting the dissipation_rate_form ' \
                                          'nullifies the ' + arg + ' argument.'
            if dissipation_rate is not None:
                warning_message('dissipation_rate')

            if dissipation_rate_form not in ['peters', 'Peters', 'uniform'] or \
                    (max_dissipation_rate is None and stoich_dissipation_rate is None):
                self._x = np.zeros_like(self._z)
                self._max_dissipation_rate = np.max(self._x)
                self._dissipation_rate_form = 'unspecified-set-to-0'
                # error_message = 'Flamelet specifications: you must specify either both of the dissipation_rate_form ' \
                #                 'and max_dissipation_rate or only the dissipation_rate argument'
                # raise ValueError(error_message)
            else:
                if max_dissipation_rate is not None:
                    self._max_dissipation_rate = max_dissipation_rate
                    self._dissipation_rate_form = dissipation_rate_form
                    self._x = self._compute_dissipation_rate(self._z,
                                                             self._max_dissipation_rate,
                                                             self._dissipation_rate_form)
                elif stoich_dissipation_rate is not None:
                    if dissipation_rate_form in ['peters', 'Peters']:
                        z_st = self.mechanism.stoich_mixture_fraction(self.fuel_stream, self.oxy_stream)
                        self._max_dissipation_rate = stoich_dissipation_rate / \
                                                     np.exp(-2. * (erfinv(2. * z_st - 1.)) ** 2)
                    elif dissipation_rate_form == 'uniform':
                        self._max_dissipation_rate = stoich_dissipation_rate
                    self._dissipation_rate_form = dissipation_rate_form
                    self._x = self._compute_dissipation_rate(self._z,
                                                             self._max_dissipation_rate,
                                                             self._dissipation_rate_form)
        self._lewis_numbers = lewis_numbers if lewis_numbers is not None else np.ones(self._n_species)

        self._use_scaled_heat_loss = use_scaled_heat_loss
        if self._use_scaled_heat_loss:
            self._T_conv = oxy_stream.T + self._z[1:-1] * (fuel_stream.T - oxy_stream.T)
            self._T_rad = oxy_stream.T + self._z[1:-1] * (fuel_stream.T - oxy_stream.T)
            zst = self._mechanism.stoich_mixture_fraction(fuel_stream, oxy_stream)
            factor = np.max(self._x) * (1. - zst) / zst
            self._h_conv *= factor
            self._h_rad *= factor

        # set up the initialization
        #
        # str - 'unreacted', 'equilibrium', 'linear-TY', 'Burke-Schumann'
        # ndarray - the interior state, useful for initializing another flamelet from an existing one
        if isinstance(initial_condition, str):
            state_interior = np.ndarray((self._nz_interior, self._n_equations))

            if initial_condition == 'unreacted':
                for i in range(self._nz_interior):
                    z = self._z[1 + i]
                    mixed_stream = self._mechanism.mix_streams([(self._oxy_stream, 1 - z),
                                                                (self._fuel_stream, z)],
                                                               basis='mass',
                                                               constant='HP')
                    state_interior[i, :] = np.hstack((mixed_stream.T, mixed_stream.Y[:-1]))
                self._initial_state = np.copy(state_interior.ravel())

            elif initial_condition == 'linear-TY':
                oxy_state = self._state_oxy
                fuel_state = self._state_fuel
                z = self._z[1:-1]
                for i in range(self._n_equations):
                    state_interior[:, i] = oxy_state[i] + (fuel_state[i] - oxy_state[i]) * z
                self._initial_state = np.copy(state_interior.ravel())

            elif initial_condition == 'equilibrium':
                for i in range(self._nz_interior):
                    z = self._z[1 + i]
                    mixed_stream = self._mechanism.mix_streams([(self._oxy_stream, 1 - z),
                                                                (self._fuel_stream, z)],
                                                               basis='mass',
                                                               constant='HP')
                    mixed_stream.equilibrate('HP')
                    state_interior[i, :] = np.hstack((mixed_stream.T, mixed_stream.Y[:-1]))
                self._initial_state = np.copy(state_interior.ravel())

            elif initial_condition == 'Burke-Schumann':
                atom_names = ['H', 'C', 'O', 'N']
                zst = self._mechanism.stoich_mixture_fraction(self._fuel_stream, self._oxy_stream)
                stmix = self._mechanism.mix_streams([(self._oxy_stream, 1 - zst), (self._fuel_stream, zst)],
                                                    'mass', 'HP')
                stat = self._mechanism._get_atoms_in_stream(stmix, atom_names)
                aw = 0.5 * stat['H']
                ad = stat['C']
                at = stat['N']
                mole_fraction_str = 'N2: ' + str(at / 2) + ', H2O: ' + str(aw)
                if 'CO2' in self._mechanism.species_names:
                    mole_fraction_str += ', CO2: ' + str(ad)

                sw = self._mechanism.stream('HPX', (stmix.H, self._pressure,
                                                    mole_fraction_str))

                Y_st = sw.Y
                Y_o = self._oxy_stream.Y
                Y_f = self._fuel_stream.Y

                for i in range(self._nz_interior):
                    z = self._z[1 + i]
                    h_mix = self._mechanism.mix_streams([(self._oxy_stream, 1 - z), (self._fuel_stream, z)],
                                                        'mass', 'HP').H

                    if z <= zst:
                        Y_mix = Y_o + z / zst * (Y_st - Y_o)
                    else:
                        Y_mix = Y_st + (z - zst) / (1. - zst) * (Y_f - Y_st)

                    mixed_stream = self._mechanism.stream('HPY', (h_mix, self.pressure, Y_mix))
                    state_interior[i, :] = np.hstack((mixed_stream.T, mixed_stream.Y[:-1]))

                self._initial_state = np.copy(state_interior.ravel())

            else:
                msg = 'Flamelet specifications: bad string argument for initial_condition\n' + \
                      '                         given: ' + initial_condition + '\n' + \
                      '                         allowable: ' + str(self._initializations)
                raise ValueError(msg)
        elif isinstance(initial_condition, np.ndarray):
            if initial_condition.size != self._n_dof:
                raise ValueError('size of initial condition is incorrect!')
            else:
                self._initial_state = np.copy(initial_condition)
        else:
            msg = 'Flamelet specifications: bad argument for initial_condition\n' + \
                  '                         must be either another Flamelet instance or a string\n' + \
                  '                         allowable strings: ' + str(self._initializations)
            raise ValueError(msg)
        self._final_state = np.zeros_like(self._initial_state)

        # set up the numerics
        self._griffon = self._mechanism.griffon
        self._include_enthalpy_flux = include_enthalpy_flux
        self._include_variable_cp = include_variable_cp
        self._rsopt = self._rates_sensitivity_option_dict[rates_sensitivity_type]
        self._stopt = self._sensitivity_transform_option_dict[sensitivity_transform_type]
        self._variable_scales = np.ones(self._n_dof)
        self._variable_scales[::self._n_equations] = 1.e3
        self._solution_times = []
        self._linear_inverse_operator = None

        self._maj_coeff_griffon = np.zeros(self._n_dof)
        self._sub_coeff_griffon = np.zeros(self._n_dof)
        self._sup_coeff_griffon = np.zeros(self._n_dof)
        self._mcoeff_griffon = np.zeros(self._nz_interior)
        self._ncoeff_griffon = np.zeros(self._nz_interior)
        self._griffon.flamelet_stencils(self._dz, self._nz_interior, self._x, 1. / self._lewis_numbers,
                                        self._maj_coeff_griffon, self._sub_coeff_griffon, self._sup_coeff_griffon,
                                        self._mcoeff_griffon, self._ncoeff_griffon)

        self._jac_nelements_griffon = int(
            self._n_equations * (self._nz_interior * self._n_equations + 2 * (self._nz_interior - 1)))
        row_indices = np.zeros(self._jac_nelements_griffon, dtype=np.int32)
        col_indices = np.zeros(self._jac_nelements_griffon, dtype=np.int32)
        self._griffon.flamelet_jac_indices(self._nz_interior, row_indices, col_indices)
        self._jac_indices_griffon = (row_indices, col_indices)

        self._block_thomas_l_values = np.zeros(self._n_equations * self._n_equations * self._nz_interior)
        self._block_thomas_d_factors = np.zeros(self._n_equations * self._n_equations * self._nz_interior)
        self._block_thomas_d_pivots = np.zeros(self._jac_nelements_griffon, dtype=np.int32)

        # cema and in situ processing setup
        self._V_stoich = self._gas.product_stoich_coeffs() - self._gas.reactant_stoich_coeffs()
        self._cema_Vmod = np.zeros((self._n_equations, self._n_reactions))
        self._cema_Vmod[1:, :] = np.copy(self._V_stoich[:-1, :])
        self._cema_Vmod[0, :] = \
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

        # some other setup
        self._delta_temperature_ignition = 400.
        self._iteration_count = None

    # ------------------------------------------------------------------------------------------------------------------
    # insitu processing methods
    # ------------------------------------------------------------------------------------------------------------------
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
            'mass fractions', 'mole fractions', 'production rates', 'heat release rate'

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
                       'enthalpy', 'energy', 'heat release rate']:
                self._enabled_insitu_processors.add(key)
            elif key == 'mass fractions' or 'mass fraction' in key:
                self._insitu_process_mass_fractions = True
                for species in self._gas.species_names:
                    self._enabled_insitu_processors.add('mass fraction ' + species)
            elif key == 'mole fractions' or 'mole fraction' in key:
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
        self._griffon.reactor_jac_isobaric(state, self._pressure,
                                           0, np.ndarray(1), 0, 0, 0, 0, 0, 0, 0, False, 0, 0, k, jac)
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
            ynm1 = state[1:]
            self._gas.TPY = state[0], self._pressure, hstack((ynm1, 1. - sum(ynm1)))
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
            state = self.state_with_bcs(state).flatten()
        if self._no_insitu_processors_enabled and saving_trajectory_data:
            self._solution_times.append(t)
            return
        else:
            if saving_trajectory_data:
                self._solution_times.append(t)

            # set state
            p = self._pressure

            neq = self._n_equations
            nzi = state.size // neq
            ndof = neq * nzi

            T = state[::neq]
            ylist = np.zeros((nzi, neq))
            for i in range(1, neq):
                ylist[:, i - 1] = state[i::neq]
            ylist[:, -1] = 1. - np.sum(ylist[:, :-1], axis=1)

            if self._insitu_process_mass_fractions:
                for i in range(self._n_species):
                    data_dict['mass fraction ' + self._gas.species_names[i]].append(ylist[:, i])

            if self._insitu_process_mole_fractions:
                x = np.zeros(ndof)
                self._griffon.flamelet_process_mole_fractions(state, nzi, x)
                for i in range(self._n_species):
                    data_dict['mole fraction ' + self._gas.species_names[i]].append(x[i::neq])

            if 'temperature' in self._enabled_insitu_processors:
                data_dict['temperature'].append(np.copy(T))

            if 'density' in self._enabled_insitu_processors:
                rho = np.zeros(nzi)
                self._griffon.flamelet_process_density(state, p, nzi, rho)
                if np.min(rho) < 1.e-14:
                    raise ValueError('density < 1.e-14 detected!')
                data_dict['density'].append(np.copy(rho))

            if 'heat release rate' in self._enabled_insitu_processors or self._insitu_process_rates:
                rates = np.zeros(ndof)
                self._griffon.flamelet_process_isobaric_reactor_rhs(state, p, nzi, rates)
                data_dict['heat release rate'].append(np.copy(rates[::neq]))
                for i in range(self._n_species):
                    name = 'production rate ' + self._gas.species_names[i]
                    if name in data_dict:
                        data_dict[name].append(np.copy(rates[i + 1::neq]))

            if 'heat capacity cv' in self._enabled_insitu_processors:
                cv = np.zeros(nzi)
                self._griffon.flamelet_process_cv(state, nzi, cv)
                data_dict['heat capacity cv'].append(np.copy(cv))

            if 'heat capacity cp' in self._enabled_insitu_processors:
                cp = np.zeros(nzi)
                self._griffon.flamelet_process_cp(state, nzi, cp)
                data_dict['heat capacity cp'].append(np.copy(cp))

            if 'enthalpy' in self._enabled_insitu_processors:
                h = np.zeros(nzi)
                self._griffon.flamelet_process_enthalpy(state, nzi, h)
                data_dict['enthalpy'].append(np.copy(h))

            if 'energy' in self._enabled_insitu_processors:
                e = np.zeros(nzi)
                self._griffon.flamelet_process_energy(state, nzi, e)
                data_dict['energy'].append(np.copy(e))

            # handle general cantera processors
            if len(self._enabled_cantera_insitu_processors):
                datadict = {}
                for label, _, _ in self._enabled_cantera_insitu_processors:
                    datadict[label] = np.zeros(nzi)
                for i in range(nzi):
                    self._gas.TPY = T[i], p, ylist[i, :]
                    for label, method, index in self._enabled_cantera_insitu_processors:
                        if index is None:
                            datadict[label][i] = getattr(self._gas, method)
                        else:
                            if isinstance(index, str):
                                datadict[label][i] = getattr(self._gas, method)[self._gas.species_index(index)]
                            else:
                                datadict[label][i] = getattr(self._gas, method)[index]
                for label, _, _ in self._enabled_cantera_insitu_processors:
                    data_dict[label].append(datadict[label])

            # handle chemical explosive mode analysis
            if self._cema_eigenvalue:
                lexp1 = np.zeros(nzi)
                lexp2 = np.zeros(nzi)
                ei1 = np.zeros((nzi, neq))
                ei2 = np.zeros((nzi, neq))
                nr = self._gas.n_reactions
                pi1 = np.zeros((nzi, nr))
                pi2 = np.zeros((nzi, nr))
                for i in range(nzi):
                    lexp1[i], lexp2[i], ei1[i], ei2[i], pi1[i], pi2[i] = self._do_cema_on_one_state(
                        state[i * neq:(i + 1) * neq])
                data_dict['cema-lexp1'].append(lexp1)
                if self._cema_explosion_indices:
                    for idx, name in enumerate(['T'] + self._gas.species_names[:-1]):
                        data_dict['cema-ei1 ' + name].append(np.copy(ei1[:, idx]))
                if self._cema_participation_indices:
                    for rxn_index in range(self._gas.n_reactions):
                        data_dict['cema-pi1 ' + str(rxn_index)].append(np.copy(pi1[:, rxn_index]))

                if self._cema_secondary_mode:
                    data_dict['cema-lexp2'].append(lexp2)
                    if self._cema_explosion_indices:
                        for idx, name in enumerate(['T'] + self._gas.species_names[:-1]):
                            data_dict['cema-ei2 ' + name].append(np.copy(ei2[:, idx]))
                    if self._cema_participation_indices:
                        for rxn_index in range(self._gas.n_reactions):
                            data_dict['cema-pi2 ' + str(rxn_index)].append(np.copy(pi2[:, rxn_index]))

    def _initialize_insitu_processing(self):
        if not len(self._enabled_insitu_processors) and not len(self._enabled_cantera_insitu_processors):
            self._no_insitu_processors_enabled = True
            self._do_insitu_processing(0., self.initial_interior_state, self._insitu_processed_data)
        else:
            self._no_insitu_processors_enabled = False
            for pp in self._enabled_insitu_processors:
                self._insitu_processed_data[pp] = []
            for label, method, index in self._enabled_cantera_insitu_processors:
                self._insitu_processed_data[label] = []

            self._do_insitu_processing(0., self.initial_interior_state, self._insitu_processed_data)

    def process_quantities_on_state(self, state, quantities=None):
        """Compute the specified in situ quantities on a given flamelet state.
        This will return a dictionary that maps each quantity's name to its computed values for the state.

        Parameters
        ----------
        state : np.ndarray
            the flamelet state - get this from a Flamelet object
        """
        self._no_insitu_processors_enabled = False
        if quantities is not None:
            self.insitu_process_quantity(quantities)
        data_dict = dict()
        for pp in self._enabled_insitu_processors:
            data_dict[pp] = []
        for label, method, index in self._enabled_cantera_insitu_processors:
            data_dict[label] = []
        self._do_insitu_processing(0., state, data_dict)
        for key in data_dict:
            data_dict[key] = array(data_dict[key])
        return data_dict

    # ------------------------------------------------------------------------------------------------------------------

    def _interpolate_state(self, state, new_grid):
        state_rect = np.vstack((self._state_oxy,
                                state.reshape((self._nz_interior, self._n_equations)),
                                self._state_fuel))
        for i in range(self._n_equations):
            state_rect[:, i] = interp1d(self._z, state_rect[:, i], kind='cubic')(new_grid)
        return state_rect[1:-1, :].ravel()

    def check_ignition_delay(self, state):
        ne = self._n_equations
        has_ignited = np.max(state[::ne] - self._initial_state[::ne]) > self._delta_temperature_ignition
        return has_ignited

    def _stop_at_ignition(self, state, t, nt, residual):
        has_ignited = self.check_ignition_delay(state)
        is_not_steady = residual > self._minimum_allowable_residual_for_ignition
        if is_not_steady:
            return not has_ignited
        else:
            return False

    def _stop_at_steady_after_ignition(self, state, t, nt, residual):
        has_ignited = self.check_ignition_delay(state)
        is_steady = residual < self._steady_tolerance
        return not (has_ignited and is_steady)

    def _stop_at_linear_temperature_or_steady(self, state, t, nt, residual):
        T_bc_max = max([self._oxy_stream.T, self._fuel_stream.T])
        is_linear_enough = np.max(state) < (1. + self._temperature_tolerance) * T_bc_max
        # TminusTlinear = state[::self._n_equations] - self.linear_temperature[1:-1]
        # is_linear_enough = np.max(np.abs(TminusTlinear)) < self._temperature_tolerance * T_bc_max
        is_steady = residual < self._steady_tolerance
        return not (is_linear_enough or is_steady)

    # ------------------------------------------------------------------------------------------------------------------
    # adiabatic methods
    # ------------------------------------------------------------------------------------------------------------------
    def _adiabatic_rhs(self, t, state_interior):
        rhs = np.zeros(self._n_dof)
        null = np.zeros(1)
        self._griffon.flamelet_rhs(state_interior, self._pressure,
                                   self._state_oxy, self._state_fuel,
                                   True, null, null, null, null,
                                   self._nz_interior,
                                   self._maj_coeff_griffon,
                                   self._sub_coeff_griffon,
                                   self._sup_coeff_griffon,
                                   self._mcoeff_griffon,
                                   self._ncoeff_griffon,
                                   self._x,
                                   self._include_enthalpy_flux,
                                   self._include_variable_cp,
                                   self._use_scaled_heat_loss,
                                   rhs)
        return rhs

    def _adiabatic_jac(self, state_interior):
        values = np.zeros(self._jac_nelements_griffon)
        null = np.zeros(1)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        True, null, null, null, null,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        False,
                                        0.,
                                        False,
                                        0.,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        False,
                                        null,
                                        values)
        return values

    def _adiabatic_jac_offset_scaled(self, state_interior, prefactor):
        values = np.zeros(self._jac_nelements_griffon)
        null = np.zeros(1)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        True, null, null, null, null,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        False,
                                        0.,
                                        True,
                                        prefactor,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        False,
                                        null,
                                        values)
        return values

    def _adiabatic_jac_and_eig(self, state_interior, diffterm):
        values = np.zeros(self._jac_nelements_griffon)
        expeig = np.zeros(self._n_dof)
        null = np.zeros(1)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        True, null, null, null, null,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        True,
                                        diffterm,
                                        False,
                                        0.,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        False,
                                        expeig,
                                        values)
        return values, expeig

    def _adiabatic_jac_csc(self, state_interior):
        return csc_matrix((self._adiabatic_jac(state_interior), self._jac_indices_griffon))

    def _adiabatic_jac_and_eig_csc(self, state_interior, diffterm):
        values, expeig = self._adiabatic_jac_and_eig(state_interior, diffterm)
        return csc_matrix((values, self._jac_indices_griffon)), expeig

    def _adiabatic_setup_superlu(self, state_interior, prefactor):
        jac = csc_matrix((self._adiabatic_jac_offset_scaled(state_interior, prefactor),
                          self._jac_indices_griffon))
        jac.eliminate_zeros()
        self._linear_inverse_operator = superlu_factor(jac)

    def _adiabatic_setup_block_thomas(self, state_interior, prefactor):
        self._jacobian_values = self._adiabatic_jac_offset_scaled(state_interior, prefactor)
        py_btddod_full_factorize(self._jacobian_values,
                                 self._nz_interior,
                                 self._n_equations,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 self._block_thomas_d_factors)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # nonadiabatic methods
    # ------------------------------------------------------------------------------------------------------------------
    def _nonadiabatic_rhs(self, t, state_interior):
        rhs = np.zeros(self._n_dof)
        self._griffon.flamelet_rhs(state_interior, self._pressure,
                                   self._state_oxy, self._state_fuel,
                                   False, self._T_conv, self._T_rad, self._h_conv, self._h_rad,
                                   self._nz_interior,
                                   self._maj_coeff_griffon,
                                   self._sub_coeff_griffon,
                                   self._sup_coeff_griffon,
                                   self._mcoeff_griffon,
                                   self._ncoeff_griffon,
                                   self._x,
                                   self._include_enthalpy_flux,
                                   self._include_variable_cp,
                                   self._use_scaled_heat_loss,
                                   rhs)
        return rhs

    def _nonadiabatic_jac(self, state_interior):
        values = np.zeros(self._jac_nelements_griffon)
        null = np.zeros(1)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        False, self._T_conv, self._T_rad, self._h_conv, self._h_rad,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        False,
                                        0.,
                                        False,
                                        0.,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        self._use_scaled_heat_loss,
                                        null,
                                        values)
        return values

    def _nonadiabatic_jac_offset_scaled(self, state_interior, prefactor):
        values = np.zeros(self._jac_nelements_griffon)
        null = np.zeros(1)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        False, self._T_conv, self._T_rad, self._h_conv, self._h_rad,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        False,
                                        0.,
                                        True,
                                        prefactor,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        self._use_scaled_heat_loss,
                                        null,
                                        values)
        return values

    def _nonadiabatic_jac_and_eig(self, state_interior, diffterm):
        values = np.zeros(self._jac_nelements_griffon)
        expeig = np.zeros(self._n_dof)
        self._griffon.flamelet_jacobian(state_interior,
                                        self._pressure,
                                        self._state_oxy, self._state_fuel,
                                        False, self._T_conv, self._T_rad, self._h_conv, self._h_rad,
                                        self._nz_interior,
                                        self._maj_coeff_griffon,
                                        self._sub_coeff_griffon,
                                        self._sup_coeff_griffon,
                                        self._mcoeff_griffon,
                                        self._ncoeff_griffon,
                                        self._x,
                                        True,
                                        diffterm,
                                        False,
                                        0.,
                                        self._rsopt,
                                        self._stopt,
                                        self._include_enthalpy_flux,
                                        self._include_variable_cp,
                                        self._use_scaled_heat_loss,
                                        expeig,
                                        values)
        return values, expeig

    def _nonadiabatic_jac_csc(self, state_interior):
        return csc_matrix((self._nonadiabatic_jac(state_interior), self._jac_indices_griffon))

    def _nonadiabatic_jac_and_eig_csc(self, state_interior, diffterm):
        values, expeig = self._nonadiabatic_jac_and_eig(state_interior, diffterm)
        return csc_matrix((values, self._jac_indices_griffon)), expeig

    def _nonadiabatic_setup_superlu(self, state_interior, prefactor):
        jac = csc_matrix((self._nonadiabatic_jac_offset_scaled(state_interior, prefactor),
                          self._jac_indices_griffon))
        jac.eliminate_zeros()
        self._linear_inverse_operator = superlu_factor(jac)

    def _nonadiabatic_setup_block_thomas(self, state_interior, prefactor):
        self._jacobian_values = self._nonadiabatic_jac_offset_scaled(state_interior, prefactor)
        py_btddod_full_factorize(self._jacobian_values,
                                 self._nz_interior,
                                 self._n_equations,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 self._block_thomas_d_factors)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # linear solve methods
    # ------------------------------------------------------------------------------------------------------------------
    def _solve_superlu(self, residual):
        return self._linear_inverse_operator.solve(residual), 1, True

    def _solve_block_thomas(self, residual):
        solution = np.zeros(self._n_dof)
        py_btddod_full_solve(self._jacobian_values, self._block_thomas_l_values, self._block_thomas_d_pivots,
                             self._block_thomas_d_factors, residual, self._nz_interior, self._n_equations, solution)
        return solution, 1, True

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # getter methods
    # ------------------------------------------------------------------------------------------------------------------
    def trajectory_data(self, key):
        """Obtain the simulation data associated with a particular key/label (processed in situ)"""
        if key not in self._insitu_processed_data.keys():
            print('Available data:', self._insitu_processed_data.keys())
            raise ValueError('data identifier ' + str(key) + ' is not valid!')
        else:
            return array(self._insitu_processed_data[key])

    def _get_mass_fraction_with_bcs(self, key, state):
        if isinstance(key, str):
            key = self._gas.species_index(key)
        if key == self._n_species - 1:
            yn = np.ones(self._nz_interior + 2)
            for i in range(1, self._n_equations):
                yn -= state[i::self._n_equations]
            return yn
        else:
            i = key + 1
            return state[i::self._n_equations]

    def state_with_bcs(self, state):
        nzi = state.size // self._n_equations
        return np.vstack((self.oxy_state,
                          state.reshape((nzi, self._n_equations)),
                          self.fuel_state)).ravel()

    @property
    def mechanism(self):
        return self._mechanism

    @property
    def dissipation_rate(self):
        return self._x

    @property
    def mixfrac_grid(self):
        return self._z

    @property
    def oxy_state(self):
        return self._state_oxy

    @property
    def fuel_state(self):
        return self._state_fuel

    @property
    def oxy_stream(self):
        return self._oxy_stream

    @property
    def fuel_stream(self):
        return self._fuel_stream

    @property
    def pressure(self):
        return self._pressure

    @property
    def linear_temperature(self):
        """Get the linear temperature profile"""
        To = self._oxy_stream.T
        Tf = self._fuel_stream.T
        return To + (Tf - To) * self._z

    @property
    def initial_interior_state(self):
        return self._initial_state

    @property
    def initial_state(self):
        return self.state_with_bcs(self._initial_state)

    @property
    def initial_temperature(self):
        return np.hstack((self._state_oxy[0], self._initial_state[::self._n_equations], self._state_fuel[0]))

    def initial_mass_fraction(self, key):
        return self._get_mass_fraction_with_bcs(key, self.state_with_bcs(self._initial_state))

    @property
    def final_interior_state(self):
        return self._final_state

    @property
    def final_state(self):
        return self.state_with_bcs(self._final_state)

    @property
    def final_temperature(self):
        return np.hstack((self._state_oxy[0], self._final_state[::self._n_equations], self._state_fuel[0]))

    def final_mass_fraction(self, key):
        return self._get_mass_fraction_with_bcs(key, self.state_with_bcs(self._final_state))

    @property
    def solution_times(self):
        """Obtain this reactor's integration times"""
        return array(self._solution_times)

    @property
    def iteration_count(self):
        return self._iteration_count

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # time integration, nonlinear solvers, etc.
    # ------------------------------------------------------------------------------------------------------------------
    def integrate(self,
                  termination,
                  first_time_step=1.e-6,
                  max_time_step=1.e-3,
                  minimum_time_step_count=40,
                  transient_tolerance=1.e-10,
                  write_log=False,
                  log_rate=100,
                  maximum_steps_per_jacobian=1,
                  nonlinear_solve_tolerance=1.e-12,
                  linear_solver='block thomas',
                  stepper_type=ESDIRK64,
                  nlsolver_type=SimpleNewtonSolver,
                  stepcontrol_type=PIController,
                  extra_governor_args=dict(),
                  extra_stepper_args=dict(),
                  extra_nlsolver_args=dict(),
                  extra_stepcontrol_args=dict()):
        """Base method for flamelet integration

        Parameters
        ----------
        termination : Termination object as in spitfire.time.governor
            how integration is stopped - instead of calling integrate() directly, use the integrate_to_time(), integrate_to_steady(), etc. methods of this class
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported
        """

        self._initialize_insitu_processing()

        # build the integration governor and set attributes
        governor = Governor()
        governor.termination_criteria = termination
        governor.minimum_time_step_count = minimum_time_step_count
        governor.projector_setup_rate = maximum_steps_per_jacobian
        governor.do_logging = write_log
        governor.log_rate = log_rate
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
        rhs_method = getattr(self, '_' + self._heat_transfer + '_rhs')
        if linear_solver.lower() == 'superlu':
            setup_method = getattr(self, '_' + self._heat_transfer + '_setup_superlu')
            solve_method = self._solve_superlu
        elif linear_solver.lower() == 'block thomas':
            setup_method = getattr(self, '_' + self._heat_transfer + '_setup_block_thomas')
            solve_method = self._solve_block_thomas
        else:
            raise ValueError('linear solver ' + linear_solver + ' is invalid, must be ''superlu'' or ''block thomas''')

        self._final_state = governor.integrate(right_hand_side=rhs_method,
                                               projector_setup=setup_method,
                                               projector_solve=solve_method,
                                               initial_condition=self._initial_state,
                                               controller=step_controller,
                                               method=stepper)[1]

    def integrate_to_steady(self, steady_tolerance=1.e-4, **kwargs):
        """Integrate a flamelet until steady state is reached

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported
        """

        self._steady_tolerance = steady_tolerance
        self.integrate(Steady(steady_tolerance), **kwargs)

    def integrate_to_time(self, final_time, **kwargs):
        """Integrate a flamelet until it reaches a specified simulation time

        Parameters
        ----------
        final_time : float
            time at which integration ceases
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported
        """

        self.final_time = final_time
        self.integrate(FinalTime(final_time), **kwargs)

    def integrate_to_steady_after_ignition(self,
                                           steady_tolerance=1.e-4,
                                           delta_temperature_ignition=400.,
                                           **kwargs):
        """Integrate a flamelet until steady state is reached after ignition (based on temperature) has occurred.
            This is helpful in slowly-evolving systems whose initial residual may be lower than the prescribed tolerance.

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined
        delta_temperature_ignition : float
            how much the temperature of the reactor must have increased for ignition to have occurred, default is 400 K
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported
        """

        self._delta_temperature_ignition = delta_temperature_ignition
        self._steady_tolerance = steady_tolerance
        self.integrate(CustomTermination(self._stop_at_steady_after_ignition), **kwargs)

    def integrate_for_heat_loss(self, temperature_tolerance=0.05, steady_tolerance=1.e-4, **kwargs):
        """Integrate a flamelet until the temperature profile is sufficiently linear.
            This is used to generate the heat loss dimension for flamelet libraries.
            Note that this will terminate the integration if a steady state is identified,
            which may simply indicate that the heat transfer settings were insufficient to
            drive the temperature to a linear enough profile.

        Parameters
        ----------
        temperature_tolerance : float
            tolerance for termination, where max(T) <= (1 + tolerance) max(T_oxy, T_fuel)
        steady_tolerance : float
            residual tolerance below which steady state is defined
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported
        """

        self._steady_tolerance = steady_tolerance
        self._temperature_tolerance = temperature_tolerance
        self.integrate(CustomTermination(self._stop_at_linear_temperature_or_steady), **kwargs)

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
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
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
            tolerance for the nonlinear solver
        linear_solver : str
            which linear solver to use - only 'block thomas' (default, heavily recommended) or 'superlu' are supported

        Returns
        -------
            the ignition delay of the reactor, in seconds
        """

        if delta_temperature_ignition is not None:
            self._delta_temperature_ignition = delta_temperature_ignition
        self._minimum_allowable_residual_for_ignition = minimum_allowable_residual
        self._no_insitu_processors_enabled = True
        self.integrate(CustomTermination(self._stop_at_ignition), **kwargs)
        has_ignited = self.check_ignition_delay(self._final_state)
        return self._solution_times[-1] if has_ignited else np.Inf

    def temporary_alc_in_chi_max(self):
        # import matplotlib.pyplot as plt

        tol = 1.e-8
        dh = 1.e-4
        h = 0.

        q = np.copy(self._initial_state)
        p = np.copy(self._max_dissipation_rate)

        p_target = 1.e5
        a_sign = 1.

        # plt.ion()
        # fig, axarray = plt.subplots(2, 1)

        total_iter = 0
        c = 0
        while p < p_target:
            c += 1

            # a single local nonlinear problem
            # save (q0, p0) and set up the linear system solvers

            def H_fun(q, p):
                self._x = self._compute_dissipation_rate(self._z, p)
                self._griffon.flamelet_stencils(self._dz, self._nz_interior, self._x, 1. / self._lewis_numbers,
                                                self._maj_coeff_griffon, self._sub_coeff_griffon,
                                                self._sup_coeff_griffon,
                                                self._mcoeff_griffon, self._ncoeff_griffon)
                return getattr(self, '_' + self._heat_transfer + '_rhs')(0., q)

            nzi = self._nz_interior
            neq = self._n_equations

            J = np.zeros(self._jac_nelements_griffon)
            btl = self._block_thomas_l_values
            btp = self._block_thomas_d_pivots
            btf = self._block_thomas_d_factors

            q0 = np.copy(q)
            p0 = np.copy(p)

            # setting up Newton's method on this local problem requires the tangent vectors at (q0,p0)
            # set up the Hq linear solves
            J = getattr(self, '_' + self._heat_transfer + '_jac')(q0)
            py_btddod_full_factorize(J, nzi, neq, btl, btp, btf)

            q = np.copy(q0)
            p = np.copy(p0)

            # finite difference the parameter Jacobian Hp for now
            Hp = (H_fun(q0, p0 + 1.e-2) - H_fun(q0, p0)) / 1.e-2

            # compute the tangent vectors
            phi = np.zeros_like(Hp)
            v = np.zeros_like(Hp)
            w = np.zeros_like(Hp)
            py_btddod_full_solve(J, btl, btp, btf, -Hp, nzi, neq, phi)
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
                    Hq = csc_matrix((self._adiabatic_jac(q), self._jac_indices_griffon))
                    A = np.zeros((self._n_dof + 1, self._n_dof + 1))
                    A[:-1, :-1] = Hq.todense()
                    A[:-1, -1] = Hp
                    A[-1, :-1] = Nq
                    A[-1, -1] = Np
                    b = np.zeros(self._n_dof + 1)
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
            dtarget = 1.e-4
            dhmax = 10.
            dh = np.min([dhmax, dh * np.sqrt(dtarget / dqn.dot(dqn))])

            if total_iter > 100:
                if pmp0_old * (p - p0) < 0.:
                    a_sign *= -1.
                    print('switch!')
                    dtarget *= 1.e-4

            print(np.max(q), p, p - p0, dh)
            pmp0_old = p - p0
            if c == 100:
                c = 0
                # axarray[0].plot(self._z[1:-1], q[::self._n_equations])
                # axarray[0].loglog(h, 1. / p, '.')
                # axarray[1].semilogx(1. / p, np.max(q), '.')
                # plt.pause(1.e-3)
            total_iter += 1
        # plt.ioff()
        # plt.show()

    def steady_solve_newton(self,
                            initial_guess=None,
                            tolerance=1.e-6,
                            max_iterations=10,
                            max_factor_line_search=1.5,
                            max_allowed_residual=1.e6,
                            min_allowable_state_var=-1.e-6,
                            norm_order=np.Inf,
                            log_rate=100000,
                            verbose=True):
        """Use Newton's method to solve for the steady state of this flamelet.
            Note that Newton's method is unlikely to converge unless an accurate initial guess is given.

        Parameters
        ----------
        initial_guess : np.ndarray
            the initial guess - obtain this from a Flamelet
        tolerance : float
            residual tolerance below which the solution has converged
        max_iterations : int
            maximum number of iterations before failure is detected
        max_factor_line_search : float
            the maximum factor by which the residual is allowed to increase in the line search algorithm
        max_allowed_residual : float
            the maximum allowable value of the residual
        min_allowable_state_var : float
            the lowest value (negative or zero) that a state variable can take during the solution process
        norm_order : int or np.Inf
            the order of the norm used in measuring the residual
        log_rate : int
            how often a message about the solution status is written
        verbose : bool
            whether to write out the solver status (log messages) or write out failure descriptions
        """

        def verbose_print(message):
            if verbose:
                print(message)

        if initial_guess is None:
            state = np.copy(self._initial_state)
        else:
            state = np.copy(initial_guess)

        inv_dofscales = 1. / self._variable_scales

        rhs_method = getattr(self, '_' + self._heat_transfer + '_rhs')
        jac_method = getattr(self, '_' + self._heat_transfer + '_jac')

        iteration_count = 0
        out_count = 0

        res = tolerance + 1.
        rhs = rhs_method(0., state)

        nzi = self._nz_interior
        neq = self._n_equations

        evaluate_jacobian = True
        while res > tolerance and iteration_count <= max_iterations:
            iteration_count += 1
            out_count += 1

            if evaluate_jacobian:
                J = -jac_method(state)

                py_btddod_full_factorize(J, nzi, neq,
                                         self._block_thomas_l_values,
                                         self._block_thomas_d_pivots,
                                         self._block_thomas_d_factors)
                evaluate_jacobian = True

            dstate = np.zeros(self._n_dof)
            py_btddod_full_solve(J,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 self._block_thomas_d_factors,
                                 rhs, nzi, neq, dstate)

            if any(logical_or(isinf(dstate), isnan(dstate))):
                verbose_print('nan/inf detected in state update!')
                return False

            norm_rhs_old = norm(rhs * inv_dofscales, ord=norm_order)
            alpha = 1.
            rhs = rhs_method(0., state + dstate)

            if any(logical_or(isinf(rhs), isnan(rhs))):
                verbose_print('nan/inf detected in state update!')
                return False

            while norm(rhs * inv_dofscales, ord=norm_order) > max_factor_line_search * norm_rhs_old:
                alpha *= 0.5
                dstate *= alpha
                rhs = rhs_method(0., state + dstate)
            state += dstate

            res = norm(rhs * inv_dofscales, ord=norm_order)

            if res > max_allowed_residual:
                message = 'Convergence failure! Residual of {:.2e} detected, ' \
                          'exceeds the maximum allowable value of {:.2e}.'.format(res, max_allowed_residual)
                verbose_print(message)
                return False

            if np.min(state) < min_allowable_state_var:
                message = 'Convergence failure! ' \
                          'Mass fraction or temperature < ' \
                          'min_allowable_state_var detected.'.format(res, max_allowed_residual)
                verbose_print(message)
                return False

            if out_count == log_rate and verbose:
                out_count = 0
                maxT = np.max(state)
                print('   - iter {:4}, |residual| = {:7.2e}, max(T) = {:6.1f}'.format(iteration_count, res, maxT))

        self._final_state = np.copy(state)
        if iteration_count >= max_iterations:
            message = 'Convergence failure! ' \
                      'Too many iterations required, more than allowable {:}.'.format(max_iterations)
            verbose_print(message)
            return False
        else:
            self._iteration_count = iteration_count
            return True

    def steady_solve_psitc(self,
                           initial_guess=None,
                           tolerance=1.e-6,
                           max_iterations=400,
                           min_allowable_state_var=-1.e-6,
                           ds_init=1.,
                           ds_init_decrease=4.,
                           adaptive_restart=True,
                           diffusion_factor=4.,
                           global_ds=False,
                           ds_safety=0.1,
                           ds_ramp=1.1,
                           ds_max=1.e4,
                           max_factor_line_search=1.5,
                           max_allowed_residual=1.e6,
                           log_rate=100000,
                           norm_order=np.Inf,
                           _recursion_depth=0,
                           max_recursion_depth=20,
                           verbose=True):
        """Use an adaptive pseudotransient continuation method to compute the steady state of this flamelet

        Parameters
        ----------
        initial_guess : np.ndarray
            the initial guess - obtain this from a Flamelet
        tolerance : float
            residual tolerance below which the solution has converged
        max_iterations : int
            maximum number of iterations before failure is detected
        max_factor_line_search : float
            the maximum factor by which the residual is allowed to increase in the line search algorithm
        max_allowed_residual : float
            the maximum allowable value of the residual
        min_allowable_state_var : float
            the lowest value (negative or zero) that a state variable can take during the solution process
        ds_init : float
            the initial pseudo time step (default: 1.), decrease to 1e-1 or 1e-2 for more robustness
        ds_init_decrease : float
            how the initial dual time step is decreased upon failure if adaptive_restart is used (default: 4)
        adaptive_restart : bool
            whether or not the solver restarts with decreased ds_init upon failure (default: True)
        diffusion_factor : float
            how strongly diffusion is weighted in the pseudo time step adaptation (default: 4) (expert parameter)
        global_ds : bool
            whether or not to use a global pseudo time step (default: False) (setting to True not recommended)
        ds_safety : float
            the 'safety factor' in the pseudo time step adaptation (default: 0.1), increase for speed, decrease for robustness
        ds_ramp : float
            how quickly the pseudo time step is allowed to increase (default: 1.1), increase for speed, decrease for robustness
        ds_max : flaot
            maximum allowable value of the pseudo time step (default: 1.e4)
        max_recursion_depth : int
            how many adaptive restarts may be attempted
        norm_order : int or np.Inf
            the order of the norm used in measuring the residual
        log_rate : int
            how often a message about the solution status is written
        verbose : bool
            whether to write out the solver status (log messages) or write out failure descriptions
        """
        original_args = locals()
        del original_args['self']

        def verbose_print(message):
            if verbose:
                print(message)

        if initial_guess is None:
            state = np.copy(self._initial_state)
        else:
            state = np.copy(initial_guess)

        dofscales = self._variable_scales
        inv_dofscales = 1. / dofscales

        rhs_method = getattr(self, '_' + self._heat_transfer + '_rhs')

        jac_method = getattr(self, '_' + self._heat_transfer + '_jac_and_eig')

        diffterm = diffusion_factor * self._max_dissipation_rate

        ds = np.zeros(self._n_dof)
        ds[:] = ds_init
        one = np.ones(self._n_dof)

        iteration_count = 0
        out_count = 0

        res = tolerance + 1.
        rhs = rhs_method(0., state)

        nzi = self._nz_interior
        neq = self._n_equations

        evaluate_jacobian = True
        while res > tolerance and iteration_count < max_iterations:
            iteration_count += 1
            out_count += 1

            if evaluate_jacobian:
                J, expeig = jac_method(state, diffterm)

                ds = np.min(np.vstack([ds_safety / (expeig + 1.e-16),
                                       ds_ramp * ds,
                                       ds_max * one]), axis=0)
                if global_ds or iteration_count == 1:
                    ds[:] = np.min(ds)
                one_over_ds = 1. / ds

                py_btddod_scale_and_add_diagonal(J, -1., one_over_ds, 1., nzi, neq)
                py_btddod_full_factorize(J, nzi, neq,
                                         self._block_thomas_l_values,
                                         self._block_thomas_d_pivots,
                                         self._block_thomas_d_factors)
                evaluate_jacobian = True

            dstate = np.zeros(self._n_dof)
            py_btddod_full_solve(J,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 self._block_thomas_d_factors,
                                 rhs, nzi, neq, dstate)

            if any(logical_or(isinf(dstate), isnan(dstate))):
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    verbose_print('nan/inf detected in state update and solver has already restarted more than'
                                  ' max_recursion_depth number of times!')
                    return False, np.min(ds)
                else:
                    if adaptive_restart:
                        verbose_print('NaN/Inf detected in steady_state_solve_psitc! Restarting...')
                        original_args['ds_init'] /= ds_init_decrease
                        original_args['_recursion_depth'] += 1
                        return self.steady_solve_psitc(**original_args)

            norm_rhs_old = norm(rhs * inv_dofscales, ord=norm_order)
            alpha = 1.
            rhs = rhs_method(0., state + dstate)

            if any(logical_or(isinf(rhs), isnan(rhs))):
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    if verbose:
                        print('nan/inf detected in state update and solver has already restarted more than'
                              ' max_recursion_depth number of times!')
                    return False, np.min(ds)
                else:
                    if adaptive_restart:
                        verbose_print('NaN/Inf detected in steady_state_solve_psitc! Restarting...')
                        original_args['ds_init'] /= ds_init_decrease
                        original_args['_recursion_depth'] += 1
                        return self.steady_solve_psitc(**original_args)
            while norm(rhs * inv_dofscales, ord=norm_order) > max_factor_line_search * norm_rhs_old:
                alpha *= 0.5
                dstate *= alpha
                rhs = rhs_method(0., state + dstate)
            state += dstate

            res = norm(rhs * inv_dofscales, ord=norm_order)

            if res > max_allowed_residual:
                message = 'Convergence failure in steady_solve! Residual of {:.2e} detected, ' \
                          'exceeds the maximum allowable value of {:.2e}.'.format(res, max_allowed_residual)
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    verbose_print(message + ' Solver has already restarted more than' \
                                            ' max_recursion_depth number of times!')
                    return False, np.min(ds)
                else:
                    if adaptive_restart:
                        verbose_print(message + ' Restarting...')
                        original_args['ds_init'] /= ds_init_decrease
                        original_args['_recursion_depth'] += 1
                        return self.steady_solve_psitc(**original_args)

            if np.min(state) < min_allowable_state_var:
                message = 'Convergence failure in steady_solve! ' \
                          'Mass fraction or temperature < ' \
                          'min_allowable_state_var detected.'.format(res, max_allowed_residual)
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    verbose_print(message + ' Solver has already restarted more than' \
                                            ' max_recursion_depth number of times!')
                    return False, np.min(ds)
                else:
                    if adaptive_restart:
                        verbose_print(message + ' Restarting...')
                        original_args['ds_init'] /= ds_init_decrease
                        original_args['_recursion_depth'] += 1
                        return self.steady_solve_psitc(**original_args)

            if out_count == log_rate and verbose:
                out_count = 0
                print('   - iter {:4}, max L_exp = {:7.2e}, min(ds) = {:7.2e}, '
                      '|residual| = {:7.2e}, max(T) = {:6.1f}'.format(iteration_count, np.max(expeig), np.min(ds),
                                                                      res, np.max(state)))
        if iteration_count >= max_iterations:
            return False, np.min(ds)
        else:
            self._iteration_count = iteration_count
            self._final_state = np.copy(state)
            return True, np.min(ds)

    def compute_steady_state(self, tolerance=1.e-6, verbose=False):
        """Solve for the steady state of this flamelet, using a number of numerical algorithms

        This will first try Newton's method, which is fast if it manages to converge.
        If Newton's method fails, the pseudo-transient continuation (psitc) method is used.
        Finally, if both Newton's method and psitc fail, ESDIRK64 time integration with adaptive stepping is attempted.

        This is meant to be a convenient interface for common usage.
        If it fails, try utilizing each of the steady solvers on their own with special parameters specified.

        Parameters
        ----------
        tolerance : float
            residual tolerance below which the solution has converged
        verbose : bool
            whether or not to write out status and failure messages
        """

        if not self.steady_solve_newton(tolerance=tolerance, log_rate=1, verbose=verbose):
            conv, mds = self.steady_solve_psitc(tolerance=tolerance, log_rate=40, max_iterations=200,
                                                verbose=verbose)
            # if not conv:
            #     conv, mds = self.steady_solve_psitc(tolerance=tolerance, log_rate=20, max_iterations=400,
            #                                         verbose=verbose, ds_init=1.e-2, ds_max=1.e1, ds_safety=0.1,
            #                                         diffusion_factor=0.1, ds_ramp=1.05)
            if not conv:
                self.integrate_to_steady(steady_tolerance=tolerance, transient_tolerance=1.e-8, max_time_step=1.e4,
                                         write_log=verbose, log_rate=20, first_time_step=1.e-2 * mds)
