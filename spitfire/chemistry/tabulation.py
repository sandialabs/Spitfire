"""
This module contains classes and methods for building tabulated chemistry libraries
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from multiprocessing import Pool, Manager
from time import perf_counter
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet import Flamelet
import pickle as pickle

_mixture_fraction_name = 'mixture_fraction'
_dissipation_rate_name = 'dissipation_rate'
_enthalpy_defect_name = 'enthalpy_defect'
_enthalpy_offset_name = 'enthalpy_offset'

_stoich_suffix = '_stoich'


class Dimension(object):
    def __init__(self, name: str, values: np.array, structured=True):
        self._name = name
        self._values = np.copy(values)
        self._min = np.min(values)
        self._max = np.max(values)
        self._npts = values.size
        self._structured = structured
        self._grid = self._values

        if structured:
            if self._values.size != np.unique(self._values).size:
                raise ValueError(f'Error in building structured dimension "{name}"'
                                 ', duplicate values identified!')

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def npts(self):
        return self._npts

    @property
    def structured(self):
        return self._structured

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
    
    def get_dict(self):
        return {'name': self._name, 'values': self._values, 'structured': self._structured}


class Library(object):
    def __init__(self, *dimensions):
        self._dims = dict({d.name: d for d in dimensions})
        self._props = dict()
        self._dims_ordering = dict()
        for i, d in enumerate(dimensions):
            self._dims_ordering[i] = d.name

        self._structured = all([self._dims[d].structured for d in self._dims])
        unstructured = all([not self._dims[d].structured for d in self._dims])
        if not self._structured and not unstructured:
            raise ValueError('Dimensions must be either all structured or all unstructured!')

        if self._structured:
            grid = np.meshgrid(*[self._dims[d].values for d in self._dims], indexing='ij')
            self._grid_shape = grid[0].shape
            for i, d in enumerate(self._dims):
                self._dims[d].grid = grid[i]
        else:
            dim0 = self._dims[next(self._dims.keys())]
            self._grid_shape = dim0.grid.shape
            if not all([self._dims[d].grid.shape == self._grid_shape for d in self._dims]):
                raise ValueError('Unstructured dimensions did not have the same grid shape!')

            if not all([len(self._dims[d].grid.shape) == 1 for d in self._dims]):
                raise ValueError('Unstructured dimensions were not given as flat arrays!')

        for d in self._dims:
            self.__dict__[self._dims[d].name] = d
            for a in self._dims[d].__dict__:
                self.__dict__[self._dims[d].name + a] = self._dims[d].__dict__[a]

    @classmethod
    def load_from_file(cls, file_name):
        with open(file_name, 'rb') as file_input:
            instance_dict = pickle.load(file_input)
            ordered_dims = list([None] * len(instance_dict['dimensions'].keys()))
            for index in instance_dict['dim_ordering']:
                name = instance_dict['dim_ordering'][index]
                d = instance_dict['dimensions'][name]
                ordered_dims[index] = Dimension(d['name'], d['values'], d['structured'])
            l = Library(*ordered_dims)
            for prop in instance_dict['properties']:
                l[prop] = instance_dict['properties'][prop]
            return l

    def __setitem__(self, quantity: str, values: np.ndarray):
        self._props[quantity] = np.copy(values)

    def __getitem__(self, quantity: str):
        return self._props[quantity]

    @property
    def props(self):
        return list(self._props.keys())

    @property
    def dims(self):
        dims = []
        for d in self._dims_ordering:
            dims.append(self._dims[self._dims_ordering[d]])
        return dims

    def dim(self, name):
        return self._dims[name]

    def get_empty_dataset(self):
        return np.ndarray(self._grid_shape)

    def remove_from_dataset(self, *quantities):
        for quantity in quantities:
            self._props.pop(quantity)

    def save_to_file(self, file_name):
        instance_dict = dict({'dimensions': {d: self._dims[d].get_dict() for d in self._dims}, 'dim_ordering': self._dims_ordering, 'properties': self._props})
        with open(file_name, 'wb') as file_output:
            pickle.dump(instance_dict, file_output)


class PostProcessor(object):
    def __init__(self, dependencies):
        self._dependencies = dependencies

    @property
    def dependencies(self):
        return self._dependencies

    def evaluate(self, library):
        raise TypeError('Base class PostProcessor evaluate() method was called, must be overridden in derived classes')


def build_unreacted_library(flamelet_specs,
                            tabulated_quantities,
                            verbose=True,
                            post_processors=None):
    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': 'unreacted'})
    cput00 = perf_counter()
    flamelet = Flamelet(**fs0)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building unreacted library (linear mass fractions and enthalpy)')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {flamelet.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))
    flamelet.insitu_process_quantity(tabulated_quantities)
    data_dict = flamelet.process_quantities_on_state(flamelet.initial_state)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    library = Library(z_dim)
    for quantity in tabulated_quantities:
        library[quantity] = data_dict[quantity].ravel()

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    return library


def build_adiabatic_eq_library(flamelet_specs,
                               tabulated_quantities,
                               verbose=True,
                               post_processors=None):
    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': 'equilibrium'})
    cput00 = perf_counter()
    flamelet = Flamelet(**fs0)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building adiabatic equilibrium library')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {flamelet.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))
    flamelet.insitu_process_quantity(tabulated_quantities)
    data_dict = flamelet.process_quantities_on_state(flamelet.initial_state)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    library = Library(z_dim)
    for quantity in tabulated_quantities:
        library[quantity] = data_dict[quantity].ravel()

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    return library


def build_adiabatic_bs_library(flamelet_specs,
                               tabulated_quantities,
                               verbose=True,
                               post_processors=None):
    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': 'Burke-Schumann'})
    cput00 = perf_counter()
    flamelet = Flamelet(**fs0)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building adiabatic Burke-Schumann library')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {flamelet.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))

    flamelet.insitu_process_quantity(tabulated_quantities)
    data_dict = flamelet.process_quantities_on_state(flamelet.initial_state)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    library = Library(z_dim)
    for quantity in tabulated_quantities:
        library[quantity] = data_dict[quantity].ravel()

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    return library


def _enthalpy_defect_fz(z, z_st):
    f = z.copy()
    f[z <= z_st] = z[z <= z_st] / z_st
    f[z > z_st] = (1 - z[z > z_st]) / (1 - z_st)
    return f


def _get_enthalpy(h_ad, z, z_st, d_st):
    return h_ad + d_st * _enthalpy_defect_fz(z, z_st)


def _get_defect_extremum(state_ad, h_ad, z_st, flamelet):
    ns = flamelet.mechanism.n_species
    z = flamelet.mixfrac_grid
    T_ad = state_ad[::ns]
    T_ur = z_st * T_ad[-1] + (1 - z_st) * T_ad[0]
    state_cooled_eq = state_ad.copy()
    state_cooled_eq[::ns] = T_ur
    enthalpy_cooled_eq = flamelet.process_quantities_on_state(state_cooled_eq)['enthalpy'].ravel()
    h_ad_st = interp1d(z, h_ad)(z_st)
    h_ce_st = interp1d(z, enthalpy_cooled_eq)(z_st)
    return h_ad_st - h_ce_st


def build_nonadiabatic_defect_eq_library(flamelet_specs,
                                         tabulated_quantities,
                                         n_defect_st=16,
                                         verbose=True,
                                         post_processors=None):
    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': 'equilibrium'})
    cput00 = perf_counter()
    flamelet = Flamelet(**fs0)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building nonadiabatic equilibrium library')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {flamelet.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))
    flamelet.insitu_process_quantity(tabulated_quantities)

    z = flamelet.mixfrac_grid
    ns = flamelet.mechanism.n_species

    nonad_eq_fs = dict(flamelet_specs)
    nonad_eq_fs['initial_condition'] = 'equilibrium'
    flamelet2 = Flamelet(**nonad_eq_fs)
    flamelet2.insitu_process_quantity('enthalpy')
    enthalpy_ad = flamelet2.process_quantities_on_state(flamelet2.initial_state)['enthalpy'].ravel()

    z_st = flamelet.mechanism.stoich_mixture_fraction(fuel, oxy)

    defect_ext = _get_defect_extremum(flamelet.initial_state, enthalpy_ad, z_st, flamelet2)
    defect_range = np.linspace(-defect_ext, 0, n_defect_st)[::-1]

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    g_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, defect_range)
    library = Library(z_dim, g_dim)

    for quantity in tabulated_quantities:
        library[quantity] = library.get_empty_dataset()
    library['enthalpy_defect'] = library.get_empty_dataset()
    library['enthalpy_cons'] = library.get_empty_dataset()
    library['enthalpy'] = library.get_empty_dataset()
    library[_mixture_fraction_name] = library.get_empty_dataset()

    new_state = flamelet.initial_state.copy()
    for ig in range(n_defect_st):
        new_enthalpy = _get_enthalpy(enthalpy_ad, z, z_st, defect_range[ig])
        for i in range(z.size):
            ym1 = new_state[i * ns + 1:(i + 1) * ns]
            yn = 1. - np.sum(ym1)
            this_y = np.hstack((ym1, yn))
            flamelet.mechanism.gas.HPY = new_enthalpy[i], flamelet.pressure, this_y
            flamelet.mechanism.gas.equilibrate('HP')
            new_state[i * ns] = flamelet.mechanism.gas.T
            new_state[i * ns + 1:(i + 1) * ns] = flamelet.mechanism.gas.Y[:-1]
        data_dict = flamelet.process_quantities_on_state(new_state)

        for quantity in tabulated_quantities:
            library[quantity][:, ig] = data_dict[quantity].ravel()
        library['enthalpy_defect'][:, ig] = np.copy(new_enthalpy - enthalpy_ad)
        library['enthalpy_cons'][:, ig] = np.copy(enthalpy_ad)
        library['enthalpy'][:, ig] = np.copy(new_enthalpy)
        library[_mixture_fraction_name][:, ig] = np.copy(flamelet.mixfrac_grid.ravel())

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    return library


def build_nonadiabatic_defect_bs_library(flamelet_specs,
                                         tabulated_quantities,
                                         n_defect_st=16,
                                         verbose=True,
                                         post_processors=None):
    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': 'Burke-Schumann'})
    cput00 = perf_counter()
    flamelet = Flamelet(**fs0)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building nonadiabatic equilibrium library')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {flamelet.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))
    flamelet.insitu_process_quantity(tabulated_quantities)

    z = flamelet.mixfrac_grid
    ns = flamelet.mechanism.n_species

    nonad_eq_fs = dict(flamelet_specs)
    nonad_eq_fs['initial_condition'] = 'Burke-Schumann'
    flamelet2 = Flamelet(**nonad_eq_fs)
    flamelet2.insitu_process_quantity('enthalpy')
    enthalpy_ad = flamelet2.process_quantities_on_state(flamelet2.initial_state)['enthalpy'].ravel()

    z_st = flamelet.mechanism.stoich_mixture_fraction(fuel, oxy)

    defect_ext = _get_defect_extremum(flamelet.initial_state, enthalpy_ad, z_st, flamelet2)
    defect_range = np.linspace(-defect_ext, 0, n_defect_st)[::-1]

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    g_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, defect_range)
    library = Library(z_dim, g_dim)

    for quantity in tabulated_quantities:
        library[quantity] = library.get_empty_dataset()
    library['enthalpy_defect'] = library.get_empty_dataset()
    library['enthalpy_cons'] = library.get_empty_dataset()
    library['enthalpy'] = library.get_empty_dataset()
    library[_mixture_fraction_name] = library.get_empty_dataset()

    new_state = flamelet.initial_state.copy()
    for ig in range(n_defect_st):
        new_enthalpy = get_enthalpy(enthalpy_ad, z, z_st, defect_range[ig])
        for i in range(z.size):
            ym1 = new_state[i * ns + 1:(i + 1) * ns]
            yn = 1. - np.sum(ym1)
            this_y = np.hstack((ym1, yn))
            flamelet.mechanism.gas.HPY = new_enthalpy[i], flamelet.pressure, this_y
            new_state[i * ns] = flamelet.mechanism.gas.T
            new_state[i * ns + 1:(i + 1) * ns] = flamelet.mechanism.gas.Y[:-1]
        data_dict = flamelet.process_quantities_on_state(new_state)

        for quantity in tabulated_quantities:
            library[quantity][:, ig] = data_dict[quantity].ravel()
            library['enthalpy_defect'][:, ig] = np.copy(new_enthalpy - enthalpy_ad)
            library['enthalpy_cons'][:, ig] = np.copy(enthalpy_ad)
            library['enthalpy'][:, ig] = np.copy(new_enthalpy)
        library[_mixture_fraction_name][:, ig] = np.copy(flamelet.mixfrac_grid.ravel())

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    return library


def build_adiabatic_slfm_library(flamelet_specs,
                                 tabulated_quantities,
                                 diss_rate_values=np.logspace(-3, 2, 16),
                                 diss_rate_ref='stoichiometric',
                                 verbose=True,
                                 solver_verbose=False,
                                 return_intermediates=False,
                                 post_processors=None,
                                 solver_use_psitc=True):
    """Build an adiabatic steady laminar flamelet library, tabulating quantities over the mixture fraction and dissipation rate

    Parameters
    ----------
    flamelet_specs : dict
        a dictionary of arguments for building Flamelet objects
    tabulated_quantities : list[str]
        quantities to compute over the table
    diss_rate_values : np.array
        the values of the dissipation rate at the specified reference point, defaults to logspace(-3, 2, 64)
    diss_rate_ref : str
        the reference point of the dissipation rate, either 'stoichiometric' (default) or 'maximum'
    verbose : bool
        whether or not to write out status messages of the table generation
    solver_verbose : bool
        whether or not to write out status and failure messages of the solvers driving the table
    return_intermediates : bool
        whether to return a library (False, default) or intermediates useful in building nonadiabatic libraries
    """

    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))

    m = flamelet_specs['mech_spec']
    fuel = flamelet_specs['fuel_stream']
    oxy = flamelet_specs['oxy_stream']
    fs0 = dict(flamelet_specs)
    init = 'Burke-Schumann' if 'initial_condition' not in flamelet_specs else flamelet_specs['initial_condition']
    fs0.update({'max_dissipation_rate': 0., 'initial_condition': init})
    f = Flamelet(**fs0)

    table_dict = dict()

    cput00 = perf_counter()
    if verbose:
        print('----------------------------------------------------------------------------------')
        print('building adiabatic SLFM library')
        print('----------------------------------------------------------------------------------')
        print(f'- mechanism: {m.mech_xml_path}')
        print(f'- {m.n_species} species, {m.n_reactions} elementary reactions')
        print('- tabulated quantities: ' + str(tabulated_quantities))
        print(f'- mixture fraction # of values: {f.mixfrac_grid.size}')
        print(f'- stoichiometric mixture fraction: {m.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('----------------------------------------------------------------------------------')

    nchi = diss_rate_values.size
    if verbose:
        print(f'- {diss_rate_ref} dissipation rate lower limit: {np.min(diss_rate_values):.3e} Hz')
        print(f'- {diss_rate_ref} dissipation rate upper limit: {np.max(diss_rate_values):.3e} Hz')
        print(f'- {diss_rate_ref} dissipation rate # of values: {nchi}')
        print('----------------------------------------------------------------------------------')

    diss_rate_key = 'max_dissipation_rate' if diss_rate_ref == 'maximum' else 'stoich_dissipation_rate'
    suffix = _stoich_suffix if diss_rate_ref == 'stoichiometric' else '_max'

    fs = dict(flamelet_specs)
    fs['initial_condition'] = 'equilibrium'
    x_values = list()
    for idx, chival in enumerate(diss_rate_values):
        fs[diss_rate_key] = chival
        flamelet = Flamelet(**fs)
        if verbose:
            print(f'{idx + 1:4}/{nchi:4} (chi{suffix} = {chival:8.1e} 1/s) ', end='', flush=True)
        cput0 = perf_counter()
        flamelet.compute_steady_state(tolerance=1.e-6, verbose=solver_verbose, use_psitc=solver_use_psitc)
        dcput = perf_counter() - cput0

        if np.max(flamelet.final_temperature - flamelet.linear_temperature) < 10.:
            if verbose:
                print(' extinction detected, stopping. The extinguished state will not be included in the table.')
            break
        else:
            if verbose:
                print(f' converged in {dcput:6.2f} s, T_max = {np.max(flamelet.final_state):6.1f}')

        flamelet.insitu_process_quantity(tabulated_quantities)
        data_dict = flamelet.process_quantities_on_state(flamelet.final_state)

        z_st = flamelet.mechanism.stoich_mixture_fraction(flamelet.fuel_stream, flamelet.oxy_stream)
        chi_st = float(interp1d(flamelet.mixfrac_grid, flamelet.dissipation_rate)(z_st))
        x_values.append(chi_st)

        table_dict[chi_st] = dict()
        for k in tabulated_quantities:
            table_dict[chi_st][k] = data_dict[k].ravel()
        fs['initial_condition'] = flamelet.final_interior_state
        if return_intermediates:
            table_dict[chi_st]['adiabatic_state'] = np.copy(flamelet.final_interior_state)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    if return_intermediates:
        return table_dict, f.mixfrac_grid, np.array(x_values)
    else:
        z_dim = Dimension(_mixture_fraction_name, f.mixfrac_grid)
        x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, np.array(x_values))

        library = Library(z_dim, x_dim)

        for quantity in table_dict[x_values[0]]:
            values = library.get_empty_dataset()

            for ix, x in enumerate(x_values):
                values[:, ix] = table_dict[x][quantity]

            library[quantity] = values

        if post_processors is not None:
            for p in post_processors:
                library = p.evaluate(library)
            for dep in dependency_only_quantities:
                library.remove_from_dataset(dep)

        return library


def _expand_enthalpy_defect_dimension(args):
    chi_st = args[0]
    managed_dict = args[1]
    flamelet_specs = args[2]
    mech_args = args[3]
    oxy_tpy = args[4]
    fuel_tpy = args[5]
    table_dict = args[6]
    tabulated_quantities = args[7]
    h_stoich_spacing = args[8]
    verbose = args[9]
    input_integration_args = args[10]
    solver_verbose= args[11]
    fsnad = dict(flamelet_specs)
    fsnad['mech_spec'] = ChemicalMechanismSpec(*mech_args)
    fsnad['oxy_stream'] = fsnad['mech_spec'].stream('TPY', oxy_tpy)
    fsnad['fuel_stream'] = fsnad['mech_spec'].stream('TPY', fuel_tpy)
    fsnad['initial_condition'] = table_dict[chi_st]['adiabatic_state']
    fsnad['stoich_dissipation_rate'] = chi_st
    fsnad['heat_transfer'] = 'nonadiabatic'
    fsnad['use_scaled_heat_loss'] = True
    fsnad['convection_coefficient'] = 1.e7
    fsnad['radiative_emissivity'] = 0.

    integration_args = dict({'first_time_step': 1.e-9, 'max_time_step': 1.e-1, 'write_log': solver_verbose, 'log_rate': 100})

    if input_integration_args is not None:
        integration_args.update(input_integration_args)

    fnonad = Flamelet(**fsnad)
    fnonad.insitu_process_quantity(tabulated_quantities + ['enthalpy'])
    cput0000 = perf_counter()
    fnonad.integrate_for_heat_loss(**integration_args)
    indices = [0]
    z = fnonad.mixfrac_grid
    z_st = fnonad.mechanism.stoich_mixture_fraction(fnonad.fuel_stream, fnonad.oxy_stream)
    h_tz = fnonad.trajectory_data('enthalpy')
    h_ad = h_tz[0, :]
    nt, nz = h_tz.shape
    last_hst = interp1d(z, h_ad)(z_st)
    for i in range(nt):
        this_hst = interp1d(z, h_tz[i, :])(z_st)
        if last_hst - this_hst > h_stoich_spacing:
            indices.append(i)
            last_hst = this_hst
    if nt - 1 not in indices:
        indices.append(-1)

    for i in indices:
        defect = h_tz[i, :] - h_ad
        gst = float(interp1d(z, defect)(z_st))
        this_data = dict()
        this_data['enthalpy_defect'] = np.copy(defect)
        this_data['enthalpy_cons'] = np.copy(h_ad)
        this_data['enthalpy'] = np.copy(h_tz[i, :])
        for q in tabulated_quantities:
            this_data[q] = np.copy(fnonad.trajectory_data(q)[i, :])
        this_data[_mixture_fraction_name] = fnonad.mixfrac_grid
        managed_dict[(chi_st, gst)] = this_data

    dcput = perf_counter() - cput0000

    if verbose:
        print('chi_st = {:8.1e} 1/s converged in {:6.2f} s'.format(chi_st, dcput), flush=True)


def _build_unstructured_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                         tabulated_quantities,
                                                         diss_rate_values=np.logspace(-3, 2, 16),
                                                         diss_rate_ref='stoichiometric',
                                                         verbose=True,
                                                         solver_verbose=False,
                                                         h_stoich_spacing=10.e3,
                                                         num_procs=1,
                                                         integration_args=None,
                                                         solver_use_psitc=True):
    table_dict, z_values, x_values = build_adiabatic_slfm_library(flamelet_specs, tabulated_quantities,
                                                                  diss_rate_values, diss_rate_ref,
                                                                  verbose, solver_verbose,
                                                                  return_intermediates=True,
                                                                  solver_use_psitc=solver_use_psitc)

    if verbose:
        print('integrating enthalpy defect dimension ...', flush=True)
    if num_procs > 1:
        pool = Pool(num_procs)
        manager = Manager()
        nonad_table_dict = manager.dict()
        flamelet_specs_no_ct = dict(flamelet_specs)
        del flamelet_specs_no_ct['mech_spec']
        del flamelet_specs_no_ct['fuel_stream']
        del flamelet_specs_no_ct['oxy_stream']
        mech_args = [flamelet_specs['mech_spec'].mech_xml_path, flamelet_specs['mech_spec'].group_name]
        oxy_tpy = flamelet_specs['oxy_stream'].TPY
        fuel_tpy = flamelet_specs['fuel_stream'].TPY
        cput000 = perf_counter()
        pool.map(_expand_enthalpy_defect_dimension, ((chi_st,
                                                      nonad_table_dict,
                                                      flamelet_specs_no_ct,
                                                      mech_args,
                                                      oxy_tpy,
                                                      fuel_tpy,
                                                      table_dict,
                                                      tabulated_quantities,
                                                      h_stoich_spacing,
                                                      verbose,
                                                      integration_args,
                                                      solver_verbose) for chi_st in table_dict.keys()))

        if verbose:
            print('----------------------------------------------------------------------------------')
            print('enthalpy defect dimension integrated in {:6.2f} s'.format(perf_counter() - cput000), flush=True)
            print('collecting parallel data ... '.format(perf_counter() - cput000), end='', flush=True)

        cput0000 = perf_counter()
        serial_dict = dict()
        for cg in nonad_table_dict.keys():
            serial_dict[cg] = nonad_table_dict[cg]

        if verbose:
            print('done in {:6.2f} s'.format(perf_counter() - cput0000))
            print('----------------------------------------------------------------------------------', flush=True)
        return serial_dict
    else:
        serial_dict = dict()
        flamelet_specs_no_ct = dict(flamelet_specs)
        del flamelet_specs_no_ct['mech_spec']
        del flamelet_specs_no_ct['fuel_stream']
        del flamelet_specs_no_ct['oxy_stream']
        mech_args = [flamelet_specs['mech_spec'].mech_xml_path, flamelet_specs['mech_spec'].group_name]
        oxy_tpy = flamelet_specs['oxy_stream'].TPY
        fuel_tpy = flamelet_specs['fuel_stream'].TPY
        cput000 = perf_counter()
        for chi_st in table_dict.keys():
            _expand_enthalpy_defect_dimension((chi_st,
                                               serial_dict,
                                               flamelet_specs_no_ct,
                                               mech_args,
                                               oxy_tpy,
                                               fuel_tpy,
                                               table_dict,
                                               tabulated_quantities,
                                               h_stoich_spacing,
                                               verbose,
                                               integration_args,
                                               solver_verbose))

        if verbose:
            print('----------------------------------------------------------------------------------')
            print('enthalpy defect dimension integrated in {:6.2f} s'.format(perf_counter() - cput000))
            print('----------------------------------------------------------------------------------', flush=True)
        return serial_dict


def _interpolate_to_structured_defect_dimension(unstructured_table, n_defect_stoich, verbose=False, extend=False):
    cput00 = perf_counter()
    min_g_st = 1.e305
    max_g_st = -1.e305

    chi_st_space = set()
    chi_to_g_list_dict = dict()

    progress_note = 10.
    progress = 10.
    if verbose:
        print('Structuring enthalpy defect dimension ... \nInitializing ...', end='', flush=True)

    for (chi_st, g_stoich) in unstructured_table.keys():
        min_g_st = np.min([min_g_st, g_stoich])
        max_g_st = np.max([max_g_st, g_stoich])
        chi_st_space.add(chi_st)
        if chi_st not in chi_to_g_list_dict:
            chi_to_g_list_dict[chi_st] = set({g_stoich})
        else:
            chi_to_g_list_dict[chi_st].add(g_stoich)

    if verbose:
        print(' Done.\nInterpolating onto structured grid ... \nProgress: 0%', end='', flush=True)
    defect_st_space = np.linspace(min_g_st, max_g_st, n_defect_stoich)
    if extend:
        defect_spacing = np.abs(defect_st_space[1] - defect_st_space[0])
        times_spacing = 2
        defect_st_space = np.linspace(min_g_st - times_spacing * defect_spacing, max_g_st,
                                      n_defect_stoich + times_spacing)
    for chi_st in chi_st_space:
        chi_to_g_list_dict[chi_st] = list(chi_to_g_list_dict[chi_st])
    chi_st_space = list(chi_st_space)

    structured_table = dict()
    for chi_idx, chi_st in enumerate(chi_st_space):
        unstruc_g_st_space = chi_to_g_list_dict[chi_st]
        unstruc_g_st_space_sorted = np.sort(unstruc_g_st_space)

        table_g0 = unstructured_table[(chi_st, unstruc_g_st_space[0])]
        nz = table_g0[list(table_g0.keys())[0]].size
        for g_st in defect_st_space:
            structured_table[(chi_st, g_st)] = dict()
            for q in table_g0:
                structured_table[(chi_st, g_st)][q] = np.zeros(nz)

        for q in table_g0.keys():
            for iz in range(nz):
                unstruc_y = np.zeros_like(unstruc_g_st_space)
                for i in range(len(unstruc_g_st_space)):
                    unstruc_y[i] = unstructured_table[(chi_st, unstruc_g_st_space_sorted[i])][q][iz]
                if extend and (q == 'enthalpy' or q == 'enthalpy_defect'):
                    yginterp = interp1d(unstruc_g_st_space_sorted, unstruc_y, kind='linear', fill_value='extrapolate')
                else:
                    yginterp = InterpolatedUnivariateSpline(unstruc_g_st_space_sorted, unstruc_y, k=1, ext='const')
                for g_st_struc in defect_st_space:
                    structured_table[(chi_st, g_st_struc)][q][iz] = yginterp(g_st_struc)
                    if q == 'density' and structured_table[(chi_st, g_st_struc)][q][iz] < 1.e-14:
                        raise ValueError('density < 1.e-14 detected!')
                    if q == 'temperature' and structured_table[(chi_st, g_st_struc)][q][iz] < 1.e-14:
                        raise ValueError('temperature < 1.e-14 detected!')

        if float((chi_idx + 1) / len(chi_st_space)) * 100. > progress:
            if verbose:
                print('--{:.0f}%'.format(progress), end='', flush=True)
                progress += progress_note
    if verbose:
        print('--{:.0f}%'.format(100))
    if verbose:
        print('Structured enthalpy defect dimension built in {:6.2f} s'.format(perf_counter() - cput00), flush=True)

    return structured_table, np.array(sorted(chi_st_space)), defect_st_space[::-1]


def build_nonadiabatic_defect_slfm_library(flamelet_specs,
                                           tabulated_quantities,
                                           diss_rate_values=np.logspace(-3, 2, 16),
                                           diss_rate_ref='stoichiometric',
                                           verbose=True,
                                           solver_verbose=False,
                                           h_stoich_spacing=10.e3,
                                           num_procs=1,
                                           integration_args=None,
                                           n_defect_st=32,
                                           post_processors=None,
                                           extend_defect_dim=False,
                                           solver_use_psitc=True):
    if post_processors is not None:
        dependency_only_quantities = []
        for p in post_processors:
            for dep in p.dependencies:
                if dep not in tabulated_quantities:
                    dependency_only_quantities.append(dep)
                    tabulated_quantities.append(dep)
    tabulated_quantities = list(set(tabulated_quantities))
    cput00 = perf_counter()
    ugt = _build_unstructured_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                               tabulated_quantities,
                                                               diss_rate_values,
                                                               diss_rate_ref,
                                                               verbose,
                                                               solver_verbose,
                                                               h_stoich_spacing,
                                                               num_procs,
                                                               integration_args,
                                                               solver_use_psitc)

    structured_defect_table, x_values, g_values = _interpolate_to_structured_defect_dimension(ugt,
                                                                                              n_defect_st,
                                                                                              verbose=verbose,
                                                                                              extend=extend_defect_dim)

    key0 = list(structured_defect_table.keys())[0]
    z_values = structured_defect_table[key0][_mixture_fraction_name]

    z_dim = Dimension(_mixture_fraction_name, z_values)
    x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, x_values)
    g_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, g_values)

    library = Library(z_dim, x_dim, g_dim)

    for quantity in structured_defect_table[key0]:
        values = library.get_empty_dataset()

        for ix, x in enumerate(x_values):
            for ig, g in enumerate(g_values):
                values[:, ix, ig] = structured_defect_table[(x, g)][quantity]

        library[quantity] = values

    if post_processors is not None:
        for p in post_processors:
            library = p.evaluate(library)
        for dep in dependency_only_quantities:
            library.remove_from_dataset(dep)

    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput00:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)

    return library
