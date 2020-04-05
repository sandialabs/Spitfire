"""
This module contains the Flamelet class that provides a high-level interface for nonpremixed flamelets,
namely setting up models and solving both unsteady and steady flamelets
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

from spitfire.time.integrator import odesolve
from spitfire.time.methods import KennedyCarpenterS6P4Q3
from spitfire.time.nonlinear import SimpleNewtonSolver
from spitfire.time.stepcontrol import PIController
from spitfire.chemistry.library import Dimension, Library
import numpy as np
from numpy import array
from numpy import any, logical_or, isinf, isnan
from scipy.special import erfinv
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu as superlu_factor
from numpy.linalg import norm
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
        the initial state of the flamelet, either 'equilibrium', 'unreacted', 'linear-TY', 'Burke-Schumann',
        or the interior state vector from another flamelet (obtained with Flamelet.*_interior_state properties)
    pressure : float
        the thermodynamic pressure
    oxy_stream : Cantera.Quantity (a Spitfire stream) or Cantera.Solution object
        the oxidizer stream
    fuel_stream : Cantera.Quantity (a Spitfire stream) or Cantera.Solution object
        the fuel stream
    max_dissipation_rate : float
        the maximum dissipation rate (cannot be specified alongside stoich_dissipation_rate or dissipation_rate)
    stoich_dissipation_rate : float
        the stoichiometric dissipation rate (cannot be specified alongside max_dissipation_rate or dissipation_rate)
    dissipation_rate : np.ndarray
        the np.ndarray of dissipation rates over mixture fraction (cannot be specified with maximum or stoichiometric dissipation rate)
    dissipation_rate_form : str
        the form of dissipation rate to use if the maximum value is specified ('Peters' (default) or 'constant'),
        cannot be specified with the dissipation_rate argument
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
        how the chemical source term Jacobian is formed, either 'dense' or 'sparse' for exact formulations
        or 'no-TBAF' which ignores third-body and falloff sensitivities. The default is 'dense'.
        For large mechanisms (over 100 species) the 'sparse' formulation is far faster than 'dense',
        especially for mechanisms of more than 300 species.
    sensitivity_transform_type : str
        how the Jacobian is transformed, currently only 'exact' is supported
    initial_time : float
        the starting time point (in seconds) of the reactor, default to 0.0
    """

    _heat_transfers = ['adiabatic', 'nonadiabatic']
    _initializations = ['unreacted', 'equilibrium', 'Burke-Schumann', 'linear-TY']
    _grid_types = ['uniform', 'clustered']
    _rates_sensitivity_option_dict = {'dense': 0, 'no-TBAF': 1, 'sparse': 2}
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
                 rates_sensitivity_type='dense',
                 sensitivity_transform_type='exact',
                 include_enthalpy_flux=False,
                 include_variable_cp=False,
                 use_scaled_heat_loss=False,
                 initial_time=0.):

        self._constructor_arguments = locals()
        del self._constructor_arguments['self']

        # process the mechanism
        self._oxy_stream = oxy_stream
        self._fuel_stream = fuel_stream
        self._pressure = pressure
        self._mechanism = mech_spec
        self._n_species = self._mechanism.n_species
        self._n_reactions = self._mechanism.n_reactions
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
                        self._max_dissipation_rate = stoich_dissipation_rate / np.exp(
                            -2. * (erfinv(2. * z_st - 1.)) ** 2)
                    elif dissipation_rate_form == 'uniform':
                        self._max_dissipation_rate = stoich_dissipation_rate
                    self._dissipation_rate_form = dissipation_rate_form
                    self._x = self._compute_dissipation_rate(self._z,
                                                             self._max_dissipation_rate,
                                                             self._dissipation_rate_form)
        self._lewis_numbers = np.ones(self._n_species)

        self._use_scaled_heat_loss = use_scaled_heat_loss
        if self._use_scaled_heat_loss:
            self._T_conv = oxy_stream.T + self._z[1:-1] * (fuel_stream.T - oxy_stream.T)
            self._T_rad = oxy_stream.T + self._z[1:-1] * (fuel_stream.T - oxy_stream.T)
            zst = self._mechanism.stoich_mixture_fraction(fuel_stream, oxy_stream)
            factor = np.max(self._x) * (1. - zst) / zst
            self._h_conv *= factor
            self._h_rad *= factor

        # set up the initialization
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
        self._current_state = np.copy(self._initial_state)

        self._initial_time = np.copy(initial_time)
        self._current_time = np.copy(initial_time)

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
        self._block_thomas_d_pivots = np.zeros(self._jac_nelements_griffon, dtype=np.int32)

        self._iteration_count = None

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

    def _adiabatic_setup_superlu(self, t, state_interior, prefactor):
        jac = csc_matrix((self._adiabatic_jac_offset_scaled(state_interior, prefactor),
                          self._jac_indices_griffon))
        jac.eliminate_zeros()
        self._linear_inverse_operator = superlu_factor(jac)

    def _adiabatic_setup_block_thomas(self, t, state_interior, prefactor):
        self._jacobian_values = self._adiabatic_jac_offset_scaled(state_interior, prefactor)
        py_btddod_full_factorize(self._jacobian_values,
                                 self._nz_interior,
                                 self._n_equations,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots)

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

    def _nonadiabatic_setup_superlu(self, t, state_interior, prefactor):
        jac = csc_matrix((self._nonadiabatic_jac_offset_scaled(state_interior, prefactor),
                          self._jac_indices_griffon))
        jac.eliminate_zeros()
        self._linear_inverse_operator = superlu_factor(jac)

    def _nonadiabatic_setup_block_thomas(self, t, state_interior, prefactor):
        self._jacobian_values = self._nonadiabatic_jac_offset_scaled(state_interior, prefactor)
        py_btddod_full_factorize(self._jacobian_values,
                                 self._nz_interior,
                                 self._n_equations,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots)

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # linear solve methods
    # ------------------------------------------------------------------------------------------------------------------
    def _solve_superlu(self, residual):
        return self._linear_inverse_operator.solve(residual), 1, True

    def _solve_block_thomas(self, residual):
        solution = np.zeros(self._n_dof)
        py_btddod_full_solve(self._jacobian_values, self._block_thomas_l_values, self._block_thomas_d_pivots,
                             residual, self._nz_interior, self._n_equations, solution)
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
            key = self._mechanism.species_index(key)
        if key == self._n_species - 1:
            yn = np.ones(self._nz_interior + 2)
            for i in range(1, self._n_equations):
                yn -= state[i::self._n_equations]
            return yn
        else:
            i = key + 1
            return state[i::self._n_equations]

    def _state_2d_with_bcs(self, state):
        nzi = state.size // self._n_equations
        return np.vstack((self._oxy_state,
                          state.reshape((nzi, self._n_equations)),
                          self._fuel_state)).ravel()

    @property
    def mechanism(self):
        """Obtain the flamelet's ChemicalMechanismSpec object"""
        return self._mechanism

    @property
    def dissipation_rate(self):
        """Obtain the np.ndarray of scalar dissipation rate values"""
        return self._x

    @property
    def mixfrac_grid(self):
        """Obtain the np.ndarray of mixture fraction grid points"""
        return self._z

    @property
    def oxy_stream(self):
        """Obtain the stream associated with the oxidizer"""
        return self._oxy_stream

    @property
    def fuel_stream(self):
        """Obtain the stream associated with the fuel"""
        return self._fuel_stream

    @property
    def pressure(self):
        """Obtain the thermodynamic pressure"""
        return self._pressure

    @property
    def linear_temperature(self):
        """Obtain the np.ndarray of linear temperature values"""
        To = self._oxy_stream.T
        Tf = self._fuel_stream.T
        return To + (Tf - To) * self._z

    @property
    def initial_interior_state(self):
        """Obtain the initial interior (no boundary states) state vector of the flamelet, useful for initializing another flamelet object"""
        return self._initial_state

    @property
    def initial_state(self):
        """Obtain the initial full (interior + boundaries) state vector of the flamelet"""
        return self._state_2d_with_bcs(self._initial_state)

    @property
    def initial_temperature(self):
        """Obtain the np.ndarray of values of the temperature before solving the steady/unsteady flamelet"""
        return np.hstack((self._state_oxy[0], self._initial_state[::self._n_equations], self._state_fuel[0]))

    def initial_mass_fraction(self, key):
        """Obtain the np.ndarray of values of a particular mass fraction before solving the steady/unsteady flamelet"""
        return self._get_mass_fraction_with_bcs(key, self._state_2d_with_bcs(self._initial_state))

    @property
    def current_interior_state(self):
        """Obtain the current interior (no boundary states) state vector of the flamelet, useful for initializing another flamelet object"""
        return self._current_state

    @property
    def current_state(self):
        """Obtain the initial full (interior + boundaries) state vector of the flamelet"""
        return self._state_2d_with_bcs(self._current_state)

    @property
    def current_temperature(self):
        """Obtain the np.ndarray of values of the temperature after solving the steady/unsteady flamelet"""
        return np.hstack((self._state_oxy[0], self._current_state[::self._n_equations], self._state_fuel[0]))

    def current_mass_fraction(self, key):
        """Obtain the np.ndarray of values of a particular mass fraction after solving the steady/unsteady flamelet"""
        return self._get_mass_fraction_with_bcs(key, self._state_2d_with_bcs(self._current_state))

    @property
    def solution_times(self):
        """Obtain this reactor's integration times"""
        return array(self._solution_times)

    @property
    def iteration_count(self):
        """Obtain the number of iterations needed to solve the steady flamelet (after solving...)"""
        return self._iteration_count

    @property
    def _oxy_state(self):
        return self._state_oxy

    @property
    def _fuel_state(self):
        return self._state_fuel

    def _check_ignition_delay(self, state, delta_temperature_ignition):
        ne = self._n_equations
        has_ignited = np.max(state[::ne] - self._initial_state[::ne]) > delta_temperature_ignition
        return has_ignited

    # ------------------------------------------------------------------------------------------------------------------
    # time integration, nonlinear solvers, etc.
    # ------------------------------------------------------------------------------------------------------------------
    def integrate(self,
                  stop_at_time=None,
                  stop_at_steady=None,
                  stop_criteria=None,
                  first_time_step=1.e-6,
                  max_time_step=1.e-3,
                  minimum_time_step_count=40,
                  transient_tolerance=1.e-10,
                  write_log=False,
                  log_rate=100,
                  maximum_steps_per_jacobian=1,
                  nonlinear_solve_tolerance=1.e-12,
                  linear_solver='block thomas',
                  stepper_type=KennedyCarpenterS6P4Q3,
                  nlsolver_type=SimpleNewtonSolver,
                  stepcontrol_type=PIController,
                  extra_integrator_args=dict(),
                  extra_stepper_args=dict(),
                  extra_nlsolver_args=dict(),
                  extra_stepcontrol_args=dict(),
                  save_first_and_last_only=False):
        """Base method for flamelet integration

        Parameters
        ----------
        stop_at_time : float
            The final time to stop the simulation at
        stop_at_steady : float
            The tolerance at which a steady state is decided upon and stopped at
        stop_criteria : callable (t, state, residual, n_steps)
            Any callable that returns True when the simulation should stop
        first_time_step : float
            The time step size initially used by the time integrator
        max_time_step : float
            The maximum time step allowed by the integrator
        minimum_time_step_count : int
            The minimum number of time steps to run, (default: 40) (helpful for slowly evolving simulations, for instance those with low starting temperatures)
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
        stepper_type : spitfire.time.TimeStepper
            which (single step) stepper method to use (optional, default: ESDIRK64)
        nlsolver_type : spitfire.time.NonlinearSolver
            which nonlinear solver method to use (optional, default: SimpleNewtonSolver)
        stepcontrol_type : spitfire.time.StepControl
            which time step adaptation method to use (optional, default: PIController)
        extra_integrator_args : dict
            any extra arguments to specify to the time integrator - arguments passed to the odesolve method
        extra_stepper_args : dict
            extra arguments to specify on the spitfire.time.TimeStepper object
        extra_nlsolver_args : dict
            extra arguments to specify on the spitfire.time.NonlinearSolver object
        extra_stepcontrol_args : dict
            extra arguments to specify on the spitfire.time.StepControl object
        save_first_and_last_only : bool
            whether or not to retain all data (False, default) or only the first and last solutions
        Returns
        -------
            a library containing temperature, mass fractions, and pressure over time and mixture fraction, respectively
        """

        def post_step_callback(t, state, *args):
            state[state < 0.] = 0.
            return state

        integrator_args = {'stop_criteria': stop_criteria}
        if stop_at_time is not None:
            integrator_args.update({'stop_at_time': stop_at_time})
        if stop_at_steady is not None:
            integrator_args.update({'stop_at_steady': stop_at_steady})

        integrator_args.update(extra_integrator_args)

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

        output = odesolve(right_hand_side=rhs_method,
                          initial_state=self._current_state,
                          initial_time=self._current_time,
                          step_size=step_controller,
                          method=stepper,
                          linear_setup=setup_method,
                          linear_solve=solve_method,
                          minimum_time_step_count=minimum_time_step_count,
                          linear_setup_rate=maximum_steps_per_jacobian,
                          verbose=write_log,
                          log_rate=log_rate,
                          norm_weighting=1. / self._variable_scales,
                          post_step_callback=post_step_callback,
                          save_each_step=not save_first_and_last_only,
                          **integrator_args)

        if save_first_and_last_only:
            current_state, current_time, time_step_size = output
            self._current_state = np.copy(current_state)
            self._current_time = np.copy(current_time)
            states = np.zeros((1, current_state.size))
            states[0, :] = current_state
            t = np.zeros(1)
            t[0] = current_time
        else:
            t, states = output
            self._current_state = np.copy(states[-1, :])
            self._current_time = np.copy(t[-1])

        time_dimension = Dimension('time', t)
        mixfrac_dimension = Dimension('mixture_fraction', self._z)
        output_library = Library(time_dimension, mixfrac_dimension)

        output_library['temperature'] = output_library.get_empty_dataset()
        output_library['temperature'][:, 0] = self.oxy_stream.T
        output_library['temperature'][:, 1:-1] = states[:, ::self._n_equations]
        output_library['temperature'][:, -1] = self.fuel_stream.T

        output_library['pressure'] = np.zeros_like(output_library['temperature']) + self._pressure

        species_names = self._mechanism.species_names
        output_library['mass fraction ' + species_names[-1]] = np.ones_like(output_library['temperature'])
        for i, s in enumerate(species_names[:-1]):
            output_library['mass fraction ' + s] = output_library.get_empty_dataset()
            output_library['mass fraction ' + s][:, 0] = self.oxy_stream.Y[i]
            output_library['mass fraction ' + s][:, 1:-1] = states[:, 1 + i::self._n_equations]
            output_library['mass fraction ' + s][:, -1] = self.fuel_stream.Y[i]
            output_library['mass fraction ' + species_names[-1]] -= output_library['mass fraction ' + s]

        return output_library

    def integrate_to_steady(self, steady_tolerance=1.e-4, **kwargs):
        """Integrate a flamelet until steady state is reached

        Parameters
        ----------
        steady_tolerance : float
            residual tolerance below which steady state is defined
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """
        return self.integrate(stop_at_steady=steady_tolerance, **kwargs)

    def integrate_to_time(self, final_time, **kwargs):
        """Integrate a flamelet until it reaches a specified simulation time

        Parameters
        ----------
        final_time : float
            time at which integration stops
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """
        return self.integrate(stop_at_time=final_time, **kwargs)

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
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """

        def stop_at_steady_after_ignition(t, state, residual, *args, **kwargs):
            has_ignited = self._check_ignition_delay(state, delta_temperature_ignition)
            is_steady = residual < steady_tolerance
            return has_ignited and is_steady

        return self.integrate(stop_criteria=stop_at_steady_after_ignition, **kwargs)

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
        **kwargs
            Arbitrary keyword arguments - see the integrate() method documentation
        """

        def stop_at_linear_temperature_or_steady(t, state, residual, *args, **kwargs):
            T_bc_max = max([self._oxy_stream.T, self._fuel_stream.T])
            is_linear_enough = np.max(state) < (1. + temperature_tolerance) * T_bc_max
            is_steady = residual < steady_tolerance
            return is_linear_enough or is_steady

        return self.integrate(stop_criteria=stop_at_linear_temperature_or_steady, **kwargs)

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
        """

        def stop_at_ignition(t, state, residual, *args, **kwargs):
            has_ignited = self._check_ignition_delay(state, delta_temperature_ignition)
            is_not_steady = residual > minimum_allowable_residual
            if is_not_steady:
                return has_ignited
            else:
                error_msg = f'From compute_ignition_delay(): '
                f'residual < minimum allowable value ({minimum_allowable_residual}),'
                f' suggesting that the reactor will not ignite.'
                f'\nNote that you can set this value with the "minimum_allowable_residual" argument.'
                f'\nIt is advised that you also pass write_log=True to observe progress of the simulation '
                f'in case it is running perpetually.'
                raise ValueError(error_msg)

        output_library = self.integrate(stop_criteria=stop_at_ignition,
                                        save_first_and_last_only=not return_solution,
                                        **kwargs)
        tau_ignition = output_library.time_values[-1]
        if return_solution:
            return tau_ignition, output_library
        else:
            return tau_ignition

    def _make_library_from_interior_state(self, state_in):
        mixfrac_dimension = Dimension('mixture_fraction', self._z)
        output_library = Library(mixfrac_dimension)

        output_library['temperature'] = output_library.get_empty_dataset()
        output_library['temperature'][0] = self.oxy_stream.T
        output_library['temperature'][1:-1] = state_in[::self._n_equations]
        output_library['temperature'][-1] = self.fuel_stream.T

        output_library['pressure'] = np.zeros_like(output_library['temperature']) + self._pressure

        species_names = self._mechanism.species_names
        output_library['mass fraction ' + species_names[-1]] = np.ones_like(output_library['temperature'])
        for i, s in enumerate(species_names[:-1]):
            output_library['mass fraction ' + s] = output_library.get_empty_dataset()
            output_library['mass fraction ' + s][0] = self.oxy_stream.Y[i]
            output_library['mass fraction ' + s][1:-1] = state_in[1 + i::self._n_equations]
            output_library['mass fraction ' + s][-1] = self.fuel_stream.Y[i]
            output_library['mass fraction ' + species_names[-1]] -= output_library['mass fraction ' + s]
        return output_library

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
        Returns
        -------
            a tuple of a library containing temperature, mass fractions, and pressure over mixture fraction,
            and the required iteration count, and whether or not the system converged,
            although if convergence is not obtained, then the library and iteration count output will both be None
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
        while res > tolerance and iteration_count < max_iterations:
            iteration_count += 1
            out_count += 1

            if evaluate_jacobian:
                J = -jac_method(state)

                py_btddod_full_factorize(J, nzi, neq,
                                         self._block_thomas_l_values,
                                         self._block_thomas_d_pivots)
                evaluate_jacobian = False

            dstate = np.zeros(self._n_dof)
            py_btddod_full_solve(J,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 rhs, nzi, neq, dstate)

            if any(logical_or(isinf(dstate), isnan(dstate))):
                verbose_print('nan/inf detected in state update!')
                return None, None, False

            norm_rhs_old = norm(rhs * inv_dofscales, ord=norm_order)
            alpha = 1.
            rhs = rhs_method(0., state + dstate)

            if any(logical_or(isinf(rhs), isnan(rhs))):
                verbose_print('nan/inf detected in state update!')
                return None, None, False

            while norm(rhs * inv_dofscales, ord=norm_order) > max_factor_line_search * norm_rhs_old and alpha > 0.001:
                alpha *= 0.5
                dstate *= alpha
                rhs = rhs_method(0., state + dstate)
                verbose_print(f'  line search reducing step size to {alpha:.3f}')
                evaluate_jacobian = True
            state += dstate

            res = norm(rhs * inv_dofscales, ord=norm_order)

            if res > max_allowed_residual:
                message = 'Convergence failure! Residual of {:.2e} detected, ' \
                          'exceeds the maximum allowable value of {:.2e}.'.format(res, max_allowed_residual)
                verbose_print(message)
                return None, None, False

            if np.min(state) < min_allowable_state_var:
                message = 'Convergence failure! ' \
                          'Mass fraction or temperature < ' \
                          'min_allowable_state_var detected.'.format(res, max_allowed_residual)
                verbose_print(message)
                return None, None, False

            if out_count == log_rate and verbose:
                out_count = 0
                maxT = np.max(state)
                print('   - iter {:4}, |residual| = {:7.2e}, max(T) = {:6.1f}'.format(iteration_count, res, maxT))

        state[state < 0] = 0.
        output_library = self._make_library_from_interior_state(state)

        if iteration_count > max_iterations or res > tolerance:
            message = 'Convergence failure! ' \
                      'Too many iterations required, more than allowable {:}.'.format(max_iterations)
            verbose_print(message)
            return None, None, False
        else:
            self._current_state = np.copy(state)
            return output_library, iteration_count, True

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
        Returns
        -------
            a tuple of a library containing temperature, mass fractions, and pressure over mixture fraction,
            and the required iteration count, whether or not the system converged, and the final minimum dual time step value,
            although if convergence is not obtained, then the library and iteration count output will both be None
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
        jac_age = 0
        jac_refresh_age = 8

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
                                         self._block_thomas_d_pivots)
                jac_age = 0
                evaluate_jacobian = False
            else:
                jac_age += 1

            evaluate_jacobian = (jac_age == jac_refresh_age) or res > 1.e-2

            dstate = np.zeros(self._n_dof)
            py_btddod_full_solve(J,
                                 self._block_thomas_l_values,
                                 self._block_thomas_d_pivots,
                                 rhs, nzi, neq, dstate)

            if any(logical_or(isinf(dstate), isnan(dstate))):
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    verbose_print('nan/inf detected in state update and solver has already restarted more than'
                                  ' max_recursion_depth number of times!')
                    return None, None, False, np.min(ds)
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
                    return None, None, False, np.min(ds)
                else:
                    if adaptive_restart:
                        verbose_print('NaN/Inf detected in steady_state_solve_psitc! Restarting...')
                        original_args['ds_init'] /= ds_init_decrease
                        original_args['_recursion_depth'] += 1
                        return self.steady_solve_psitc(**original_args)
            while norm(rhs * inv_dofscales, ord=norm_order) > max_factor_line_search * norm_rhs_old and alpha > 0.001:
                alpha *= 0.5
                dstate *= alpha
                rhs = rhs_method(0., state + dstate)
                verbose_print(f'  line search reducing step size to {alpha:.3f}')
                evaluate_jacobian = True
            state += dstate

            res = norm(rhs * inv_dofscales, ord=norm_order)

            if res > max_allowed_residual:
                message = 'Convergence failure in steady_solve! Residual of {:.2e} detected, ' \
                          'exceeds the maximum allowable value of {:.2e}.'.format(res, max_allowed_residual)
                if _recursion_depth > max_recursion_depth or not adaptive_restart:
                    verbose_print(message + ' Solver has already restarted more than' \
                                            ' max_recursion_depth number of times!')
                    return None, None, False, np.min(ds)
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
                    return None, None, False, np.min(ds)
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
        state[state < 0] = 0.
        output_library = self._make_library_from_interior_state(state)

        if iteration_count > max_iterations or res > tolerance:
            return None, None, False, np.min(ds)
        else:
            self._iteration_count = iteration_count
            self._current_state = np.copy(state)
            return output_library, iteration_count, True, np.min(ds)

    def compute_steady_state(self, tolerance=1.e-6, verbose=False,
                             use_psitc=True, newton_args=None, psitc_args=None, transient_args=None):
        """Solve for the steady state of this flamelet, using a number of numerical algorithms

        This will first try Newton's method, which is fast if it manages to converge.
        If Newton's method fails, the pseudo-transient continuation (psitc) method is used.
        The psitc solver will attempt several restarts with increasingly conservative solver settings.
        Finally, if both Newton's method and psitc fail, ESDIRK64 time integration with adaptive stepping is attempted.

        This is meant to be a convenient interface for common usage.
        If it fails, try utilizing each of the steady solvers on their own with special parameters specified.

        For exceptionally large mechanisms, say, > 150 species, the psitc solver can be slow,
        and setting use_psitc=False will bypass it when Newton's method fails in favor of the ESDIRK solver.
        This is only recommended for large mechanisms.

        Parameters
        ----------
        tolerance : float
            residual tolerance below which the solution has converged
        verbose : bool
            whether or not to write out status and failure messages
        use_psitc : bool
            whether or not to use the psitc method when Newton's method fails (if False, tries ESDIRK time stepping next)
        newton_args : dict
            extra arguments such as max_iterations to pass to the Newton solver
        psitc_args : dict
            extra arguments such as max_iterations to pass to the PsiTC solver
        transient_args : dict
            extra arguments such as max_iterations to pass to the ESDIRK solver
        Returns
        -------
            a library containing temperature, mass fractions, and pressure over mixture fraction
        """

        the_newton_args = {'tolerance': tolerance, 'log_rate': 1, 'verbose': verbose, 'max_iterations': 38}
        if newton_args is not None:
            the_newton_args.update(newton_args)
        output_library, iteration_count, conv = self.steady_solve_newton(**the_newton_args)

        if conv:
            return output_library
        else:
            conv = False
            mds = 1.e-6

            the_psitc_args = {'tolerance': tolerance, 'log_rate': 1, 'verbose': verbose, 'max_iterations': 200}
            if psitc_args is not None:
                the_psitc_args.update(psitc_args)
            if use_psitc:
                output_library, iteration_count, conv, mds = self.steady_solve_psitc(**the_psitc_args)
            if conv:
                return output_library

            else:
                the_esdirk_args = {'steady_tolerance': tolerance,
                                   'transient_tolerance': 1.e-8,
                                   'max_time_step': 1e4,
                                   'write_log': verbose,
                                   'log_rate': 1,
                                   'first_time_step': 1e-2 * mds,
                                   'maximum_steps_per_jacobian': 10,
                                   'save_first_and_last_only': True}
                if transient_args is not None:
                    the_esdirk_args.update(transient_args)

                transient_library = self.integrate_to_steady(**the_esdirk_args)
                steady_library = Library(transient_library.dim('mixture_fraction'))
                for p in transient_library.props:
                    steady_library[p] = transient_library[p][-1, :].ravel()
                return steady_library
