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
from spitfire.chemistry.flamelet import Flamelet, FlameletSpec
from spitfire.chemistry.library import Dimension, Library
import spitfire.chemistry.analysis as sca
import copy
from functools import partial
import itertools
from scipy import integrate

"""these names are specific to tabulated chemistry libraries for combustion"""
_mixture_fraction_name = 'mixture_fraction'
_dissipation_rate_name = 'dissipation_rate'
_enthalpy_defect_name = 'enthalpy_defect'
_enthalpy_offset_name = 'enthalpy_offset'
_scaled_scalar_variance_name = 'scaled_scalar_variance_mean'
_stoich_suffix = '_stoich'
_mean_suffix = '_mean'


def _write_library_header(lib_type, mech, fuel, oxy, verbose):
    if verbose:
        print('-' * 82)
        print(f'building {lib_type} library')
        print('-' * 82)
        print(f'- mechanism: {mech.mech_file_path}')
        print(f'- {mech.n_species} species, {mech.n_reactions} reactions')
        print(f'- stoichiometric mixture fraction: {mech.stoich_mixture_fraction(fuel, oxy):.3f}')
        print('-' * 82)
    return perf_counter()


def _write_library_footer(cput0, verbose):
    if verbose:
        print('----------------------------------------------------------------------------------')
        print(f'library built in {perf_counter() - cput0:6.2f} s')
        print('----------------------------------------------------------------------------------', flush=True)


def build_unreacted_library(flamelet_specs, verbose=True):
    """Build a flamelet library for a nonreacting flow, with linear enthalpy and mass fraction profiles.

    Parameters
    ----------
    flamelet_specs : FlameletSpec or dictionary of arguments for a FlameletSpec
        flamelet specifications

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        a chemistry library with only the "mixture_fraction" dimension

    """
    fs = FlameletSpec(**flamelet_specs) if isinstance(flamelet_specs, dict) else copy.copy(flamelet_specs)
    fs.initial_condition = 'unreacted'
    flamelet = Flamelet(fs)
    return flamelet.make_library_from_interior_state(flamelet.initial_interior_state)


def build_adiabatic_eq_library(flamelet_specs, verbose=True):
    """Build a flamelet library with the equilibrium (infinitely fast) chemistry assumption,
        equivalently with Gibbs free energy minimization.

    Parameters
    ----------
    flamelet_specs : FlameletSpec or dictionary of arguments for a FlameletSpec
        flamelet specifications

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        a chemistry library with only the "mixture_fraction" dimension

    """
    fs = FlameletSpec(**flamelet_specs) if isinstance(flamelet_specs, dict) else copy.copy(flamelet_specs)
    fs.initial_condition = 'equilibrium'
    flamelet = Flamelet(fs)
    return flamelet.make_library_from_interior_state(flamelet.initial_interior_state)


def build_adiabatic_bs_library(flamelet_specs, verbose=True):
    """Build a flamelet library with the Burke-Schumann (idealized, one-step combustion) assumptions

    Parameters
    ----------
    flamelet_specs : FlameletSpec or dictionary of arguments for a FlameletSpec
        flamelet specifications

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        a chemistry library with only the "mixture_fraction" dimension

    """
    fs = FlameletSpec(**flamelet_specs) if isinstance(flamelet_specs, dict) else copy.copy(flamelet_specs)
    fs.initial_condition = 'Burke-Schumann'
    flamelet = Flamelet(fs)
    return flamelet.make_library_from_interior_state(flamelet.initial_interior_state)


def _build_nonadiabatic_defect_unstrained_library(initialization, flamelet_specs, n_defect_st=16, verbose=True):
    flamelet_specs = FlameletSpec(**flamelet_specs) if isinstance(flamelet_specs, dict) else copy.copy(flamelet_specs)

    m = flamelet_specs.mech_spec
    fuel = flamelet_specs.fuel_stream
    oxy = flamelet_specs.oxy_stream

    z_st = m.stoich_mixture_fraction(fuel, oxy)

    flamelet_specs.initial_condition = initialization
    flamelet = Flamelet(flamelet_specs)

    # compute the extreme enthalpy defect
    state_ad = flamelet.initial_interior_state
    adiabatic_lib = flamelet.make_library_from_interior_state(state_ad)
    enthalpy_ad = sca.compute_specific_enthalpy(m, adiabatic_lib)['enthalpy']

    z_interior = flamelet.mixfrac_grid[1:-1]
    state_cooled_eq = state_ad.copy()
    state_cooled_eq[::m.n_species] = z_interior * fuel.T + (1 - z_interior) * oxy.T
    cooled_lib = flamelet.make_library_from_interior_state(state_cooled_eq)
    enthalpy_cooled_eq = sca.compute_specific_enthalpy(m, cooled_lib)['enthalpy']

    z = flamelet.mixfrac_grid
    h_ad_st = interp1d(z, enthalpy_ad)(z_st)
    h_ce_st = interp1d(z, enthalpy_cooled_eq)(z_st)
    defect_ext = h_ad_st - h_ce_st

    # build the library with equilibrium solutions at with enthalpies offset by the triangular defect form
    defect_range = np.linspace(-defect_ext, 0, n_defect_st)[::-1]
    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    g_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, defect_range)
    output_library = Library(z_dim, g_dim)
    output_library.extra_attributes['mech_spec'] = m

    for p in adiabatic_lib.props:
        output_library[p] = output_library.get_empty_dataset()
    output_library['enthalpy_defect'] = output_library.get_empty_dataset()
    output_library['enthalpy_cons'] = output_library.get_empty_dataset()
    output_library['enthalpy'] = output_library.get_empty_dataset()
    output_library[_mixture_fraction_name] = output_library.get_empty_dataset()

    fz = z.copy()
    fz[z <= z_st] = z[z <= z_st] / z_st
    fz[z > z_st] = (1 - z[z > z_st]) / (1 - z_st)

    ns = m.n_species
    g_library = flamelet.make_library_from_interior_state(flamelet.initial_interior_state)
    for ig in range(n_defect_st):
        defected_enthalpy = enthalpy_ad + defect_range[ig] * fz

        for iz in range(1, z.size - 1):
            y = np.zeros(ns)
            for ispec in range(ns):
                y[ispec] = g_library['mass fraction ' + m.species_names[ispec]][iz]
            m.gas.HPY = defected_enthalpy[iz], flamelet.pressure, y
            if initialization == 'equilibrium':
                m.gas.equilibrate('HP')
            g_library['temperature'][iz] = m.gas.T
            for ispec in range(ns):
                g_library['mass fraction ' + m.species_names[ispec]][iz] = m.gas.Y[ispec]

        for p in g_library.props:
            if p != 'defected_enthapy':
                output_library[p][:, ig] = g_library[p].ravel()
        output_library['enthalpy_defect'][:, ig] = defected_enthalpy - enthalpy_ad
        output_library['enthalpy_cons'][:, ig] = enthalpy_ad
        output_library['enthalpy'][:, ig] = defected_enthalpy
        output_library[_mixture_fraction_name][:, ig] = flamelet.mixfrac_grid.ravel()

    return output_library


def build_nonadiabatic_defect_eq_library(flamelet_specs, n_defect_st=16, verbose=True):
    """Build a flamelet library with the equilibrium (infinitely fast) chemistry assumption
        and heat loss effects captured through a presumed triangular form of the enthalpy defect.

    Parameters
    ----------
    flamelet_specs : FlameletSpec or dictionary of arguments for a FlameletSpec
        flamelet specifications
    n_defect_st : Int
        the number of stoichiometric enthalpy defect values to include in the table (default: 16)

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        a chemistry library with the "mixture_fraction" and "enthalpy_defect_stoich" dimensions

    """
    return _build_nonadiabatic_defect_unstrained_library('equilibrium', flamelet_specs, n_defect_st, verbose)


def build_nonadiabatic_defect_bs_library(flamelet_specs, n_defect_st=16, verbose=True):
    """Build a flamelet library with the Burke-Schumann chemistry assumption
        and heat loss effects captured through a presumed triangular form of the enthalpy defect.

    Parameters
    ----------
    flamelet_specs : FlameletSpec or dictionary of arguments for a FlameletSpec
        flamelet specifications
    n_defect_st : Int
        the number of stoichiometric enthalpy defect values to include in the table (default: 16)

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        a chemistry library with the "mixture_fraction" and "enthalpy_defect_stoich" dimensions

    """
    return _build_nonadiabatic_defect_unstrained_library('Burke-Schumann', flamelet_specs, n_defect_st, verbose)


def build_adiabatic_slfm_library(flamelet_specs,
                                 diss_rate_values=np.logspace(-3, 2, 16),
                                 diss_rate_ref='stoichiometric',
                                 verbose=True,
                                 solver_verbose=False,
                                 _return_intermediates=False,
                                 include_extinguished=False,
                                 diss_rate_log_scaled=True):
    """Build a flamelet library with an adiabatic strained laminar flamelet model

    Parameters
    ----------
    flamelet_specs : dictionary or FlameletSpec instance
        data for the mechanism, streams, mixture fraction grid, etc.
    diss_rate_values : np.array
        reference dissipation rate values in the table (note that if the flamelet extinguishes at any point,
        the extinguished flamelet and larger dissipation rates are not included in the library unless the
        include_extinguished argument is set to True)
    diss_rate_ref : str
        the reference point of the specified dissipation rate values, either 'stoichiometric' or 'maximum'
    verbose : bool
        whether or not to show progress of the library construction
    include_extinguished : bool
        whether or not to include extinguished states in the output table, if encountered in the provided range of
        dissipation rates, off by default
    diss_rate_log_scaled : bool
        whether or not the range of dissipation rates is logarithmically scaled 

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        the structured chemistry library

    """

    if isinstance(flamelet_specs, dict):
        flamelet_specs = FlameletSpec(**flamelet_specs)

    m = flamelet_specs.mech_spec
    fuel = flamelet_specs.fuel_stream
    oxy = flamelet_specs.oxy_stream
    flamelet_specs.initial_condition = 'equilibrium'
    if diss_rate_ref == 'maximum':
        flamelet_specs.max_dissipation_rate = 0.
    else:
        flamelet_specs.stoich_dissipation_rate = 0.

    cput00 = _write_library_header('adiabatic SLFM', m, fuel, oxy, verbose)

    f = Flamelet(flamelet_specs)

    table_dict = dict()

    nchi = diss_rate_values.size
    suffix = _stoich_suffix if diss_rate_ref == 'stoichiometric' else '_max'

    x_values = list()
    for idx, chival in enumerate(diss_rate_values):
        if diss_rate_ref == 'maximum':
            flamelet_specs.max_dissipation_rate = chival
        else:
            flamelet_specs.stoich_dissipation_rate = chival

        flamelet = Flamelet(flamelet_specs)
        if verbose:
            print(f'{idx + 1:4}/{nchi:4} (chi{suffix} = {chival:8.1e} 1/s) ', end='', flush=True)
        cput0 = perf_counter()
        x_library = flamelet.compute_steady_state(tolerance=1.e-6, verbose=solver_verbose, use_psitc=True)
        dcput = perf_counter() - cput0

        if np.max(flamelet.current_temperature - flamelet.linear_temperature) < 10. and not include_extinguished:
            if verbose:
                print(' extinction detected, stopping. The extinguished state will not be included in the table.')
            break
        else:
            if verbose:
                print(f' converged in {dcput:6.2f} s, T_max = {np.max(flamelet.current_temperature):6.1f}')

        z_st = flamelet.mechanism.stoich_mixture_fraction(flamelet.fuel_stream, flamelet.oxy_stream)
        chi_st = flamelet._compute_dissipation_rate(np.array([z_st]),
                                                    flamelet._max_dissipation_rate,
                                                    flamelet._dissipation_rate_form)[0]
        x_values.append(chi_st)

        table_dict[chi_st] = dict()
        for k in x_library.props:
            table_dict[chi_st][k] = x_library[k].ravel()
        flamelet_specs.initial_condition = flamelet.current_interior_state
        if _return_intermediates:
            table_dict[chi_st]['adiabatic_state'] = np.copy(flamelet.current_interior_state)

    if _return_intermediates:
        _write_library_footer(cput00, verbose)
        return table_dict, f.mixfrac_grid, np.array(x_values)
    else:
        z_dim = Dimension(_mixture_fraction_name, f.mixfrac_grid)
        x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, np.array(x_values), diss_rate_log_scaled)

        output_library = Library(z_dim, x_dim)
        output_library.extra_attributes['mech_spec'] = m

        for quantity in table_dict[chi_st]:
            output_library[quantity] = output_library.get_empty_dataset()
            for ix, x in enumerate(x_values):
                output_library[quantity][:, ix] = table_dict[x][quantity]

        _write_library_footer(cput00, verbose)
        return output_library


def _expand_enthalpy_defect_dimension_transient(chi_st, managed_dict, flamelet_specs, table_dict,
                                                h_stoich_spacing, verbose, input_integration_args, solver_verbose):
    flamelet_specs.initial_condition = table_dict[chi_st]['adiabatic_state']
    flamelet_specs.stoich_dissipation_rate = chi_st
    flamelet_specs.heat_transfer = 'nonadiabatic'
    flamelet_specs.scale_heat_loss_by_temp_range = True
    flamelet_specs.scale_convection_by_dissipation = True
    flamelet_specs.use_linear_ref_temp_profile = True
    flamelet_specs.convection_coefficient = flamelet_specs.convection_coefficient if flamelet_specs.convection_coefficient is not None else 1.e7
    flamelet_specs.radiative_emissivity = 0.

    integration_args = dict(
        {'first_time_step': 1.e-9,
         'max_time_step': 1.e-1,
         'write_log': solver_verbose,
         'log_rate': 100,
         'print_exception_on_failure': False})

    if input_integration_args is not None:
        integration_args.update(input_integration_args)

    if 'transient_tolerance' not in integration_args:
        integration_args['transient_tolerance'] = 1.e-8

    cput0000 = perf_counter()
    running = True
    while running and integration_args['transient_tolerance'] > 1.e-15:
        try:
            fnonad = Flamelet(flamelet_specs)
            transient_lib = fnonad.integrate_for_heat_loss(**integration_args)
            running = False
        except Exception as e:
            if solver_verbose:
                print(
                    f'Transient heat loss calculation failed with tolerance of {integration_args["transient_tolerance"]:.1e}, retrying with 100x lower...')
            integration_args.update(dict({'transient_tolerance': integration_args['transient_tolerance'] * 1.e-2}))
    indices = [0]
    z = fnonad.mixfrac_grid
    z_st = fnonad.mechanism.stoich_mixture_fraction(fnonad.fuel_stream, fnonad.oxy_stream)
    h_tz = sca.compute_specific_enthalpy(flamelet_specs.mech_spec, transient_lib)['enthalpy']
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
        this_data[_mixture_fraction_name] = fnonad.mixfrac_grid
        for q in transient_lib.props:
            this_data[q] = transient_lib[q][i, :]
        managed_dict[(chi_st, gst)] = this_data

    dcput = perf_counter() - cput0000

    if verbose:
        print('chi_st = {:8.1e} 1/s converged in {:6.2f} s'.format(chi_st, dcput), flush=True)


def _expand_enthalpy_defect_dimension_steady(chi_st, managed_dict, flamelet_specs, table_dict,
                                             h_stoich_spacing, verbose, input_integration_args, solver_verbose):
    flamelet_specs.initial_condition = table_dict[chi_st]['adiabatic_state']
    flamelet_specs.stoich_dissipation_rate = chi_st
    flamelet_specs.heat_transfer = 'nonadiabatic'
    flamelet_specs.scale_heat_loss_by_temp_range = False
    flamelet_specs.scale_convection_by_dissipation = False
    flamelet_specs.use_linear_ref_temp_profile = True
    flamelet_specs.radiative_emissivity = 0.
    flamelet_specs.convection_coefficient = 0.

    flamelet = Flamelet(flamelet_specs)

    first = True
    refine_before_extinction = False
    extinguished = False
    extinguished_first = False
    maxT = -1
    state_old = np.copy(flamelet.current_interior_state)
    hval = 0.
    dh = 1.e-1
    diff_target = 1e-1
    diff_norm = 1e-1

    hval_max = 1.e10

    solutions = []
    hvalues = []
    hvalues.append(hval)
    solutions.append(dict())

    for p in table_dict[chi_st]:
        if p != 'adiabatic_state':
            solutions[-1][p] = table_dict[chi_st][p]

    current_state = table_dict[chi_st]['adiabatic_state']

    cput0000 = perf_counter()
    while first or (not extinguished and hval < hval_max):
        hval += dh
        if first:
            first = False

        flamelet_specs.convection_coefficient = hval
        flamelet_specs.initial_condition = current_state
        flamelet = Flamelet(flamelet_specs)

        g_library = flamelet.compute_steady_state(verbose=solver_verbose)
        current_state = flamelet.current_interior_state
        maxT = np.max(current_state)

        diff_norm = np.max(np.abs(current_state - state_old) / (np.abs(current_state) + 1.e-4))

        extinguished = maxT < (np.max([flamelet.oxy_stream.T, flamelet.fuel_stream.T]) + 10.)
        if (extinguished and (not extinguished_first)) and refine_before_extinction:
            extinguished_first = True
            extinguished = False

            hval -= dh
            dh *= 0.1
            diff_target *= 0.1
            current_state = state_old.copy()

            continue

        state_old = np.copy(current_state)
        dh *= np.min([np.max([np.sqrt(diff_target / diff_norm), 0.1]), 2.])
        hvalues.append(hval)
        solutions.append(dict())
        for p in g_library.props:
            solutions[-1][p] = g_library[p].ravel()

    z_dim = Dimension(_mixture_fraction_name, flamelet.mixfrac_grid)
    h_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, np.array(hvalues))
    steady_lib = Library(z_dim, h_dim)
    steady_lib.extra_attributes['mech_spec'] = flamelet_specs.mech_spec
    for p in table_dict[chi_st]:
        if p != 'adiabatic_state':
            steady_lib[p] = steady_lib.get_empty_dataset()
    for ig, sol in enumerate(solutions):
        for p in sol:
            steady_lib[p][:, ig] = sol[p].ravel()

    indices = [0]
    z = flamelet.mixfrac_grid
    z_st = flamelet.mechanism.stoich_mixture_fraction(flamelet.fuel_stream, flamelet.oxy_stream)
    h_tz = sca.compute_specific_enthalpy(flamelet_specs.mech_spec, steady_lib)['enthalpy']
    h_ad = h_tz[:, 0]
    nz, nt = h_tz.shape
    last_hst = interp1d(z, h_ad)(z_st)
    for i in range(nt - 1):
        this_hst = interp1d(z, h_tz[:, i])(z_st)
        if last_hst - this_hst > h_stoich_spacing:
            indices.append(i)
            last_hst = this_hst

    for i in indices:
        defect = h_tz[:, i] - h_ad
        gst = float(interp1d(z, defect)(z_st))
        this_data = dict()
        this_data['enthalpy_defect'] = np.copy(defect)
        this_data['enthalpy_cons'] = np.copy(h_ad)
        this_data['enthalpy'] = np.copy(h_tz[:, i])
        this_data[_mixture_fraction_name] = flamelet.mixfrac_grid
        for q in steady_lib.props:
            this_data[q] = steady_lib[q][:, i]
        managed_dict[(chi_st, gst)] = this_data

    dcput = perf_counter() - cput0000

    if verbose:
        print('chi_st = {:8.1e} 1/s converged in {:6.2f} s'.format(chi_st, dcput), flush=True)


def _build_unstructured_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                         heat_loss_expansion='transient',
                                                         diss_rate_values=np.logspace(-3, 2, 16),
                                                         diss_rate_ref='stoichiometric',
                                                         verbose=True,
                                                         solver_verbose=False,
                                                         h_stoich_spacing=10.e3,
                                                         num_procs=1,
                                                         integration_args=None):
    table_dict, z_values, x_values = build_adiabatic_slfm_library(flamelet_specs,
                                                                  diss_rate_values, diss_rate_ref,
                                                                  verbose, solver_verbose,
                                                                  _return_intermediates=True)
    if heat_loss_expansion == 'transient':
        enthalpy_expansion_fxn = _expand_enthalpy_defect_dimension_transient
    else:
        enthalpy_expansion_fxn = _expand_enthalpy_defect_dimension_steady

    if verbose:
        print(f'expanding ({heat_loss_expansion}) enthalpy defect dimension ...', flush=True)
    if num_procs > 1:
        pool = Pool(num_procs)
        manager = Manager()
        nonad_table_dict = manager.dict()
        cput000 = perf_counter()

        pool.starmap(enthalpy_expansion_fxn,
                     ((chi_st,
                       nonad_table_dict,
                       flamelet_specs,
                       table_dict,
                       h_stoich_spacing,
                       verbose,
                       integration_args,
                       solver_verbose) for chi_st in table_dict.keys()))
        if verbose:
            print('----------------------------------------------------------------------------------')
            print('enthalpy defect dimension expanded in {:6.2f} s'.format(perf_counter() - cput000), flush=True)
            print('collecting parallel data ... '.format(perf_counter() - cput000), end='', flush=True)

        cput0000 = perf_counter()
        serial_dict = dict()
        for cg in nonad_table_dict.keys():
            serial_dict[cg] = nonad_table_dict[cg]

        pool.close()
        pool.join()

        if verbose:
            print('done in {:6.2f} s'.format(perf_counter() - cput0000))
            print('----------------------------------------------------------------------------------', flush=True)
        return serial_dict
    else:
        serial_dict = dict()
        cput000 = perf_counter()
        for chi_st in table_dict.keys():
            enthalpy_expansion_fxn(chi_st,
                                   serial_dict,
                                   flamelet_specs,
                                   table_dict,
                                   h_stoich_spacing,
                                   verbose,
                                   integration_args,
                                   solver_verbose)

        if verbose:
            print('----------------------------------------------------------------------------------')
            print('enthalpy defect dimension expanded in {:6.2f} s'.format(perf_counter() - cput000))
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


def _build_nonadiabatic_defect_slfm_library(flamelet_specs,
                                            heat_loss_expansion='transient',
                                            diss_rate_values=np.logspace(-3, 2, 16),
                                            diss_rate_ref='stoichiometric',
                                            verbose=True,
                                            solver_verbose=False,
                                            h_stoich_spacing=10.e3,
                                            num_procs=1,
                                            integration_args=None,
                                            n_defect_st=32,
                                            extend_defect_dim=False,
                                            diss_rate_log_scaled=True):
    if isinstance(flamelet_specs, dict):
        flamelet_specs = FlameletSpec(**flamelet_specs)

    m = flamelet_specs.mech_spec
    fuel = flamelet_specs.fuel_stream
    oxy = flamelet_specs.oxy_stream

    cput00 = _write_library_header('nonadiabatic (defect) SLFM', m, fuel, oxy, verbose)

    ugt = _build_unstructured_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                               heat_loss_expansion,
                                                               diss_rate_values,
                                                               diss_rate_ref,
                                                               verbose,
                                                               solver_verbose,
                                                               h_stoich_spacing,
                                                               num_procs,
                                                               integration_args)

    structured_defect_table, x_values, g_values = _interpolate_to_structured_defect_dimension(ugt,
                                                                                              n_defect_st,
                                                                                              verbose=verbose,
                                                                                              extend=extend_defect_dim)

    key0 = list(structured_defect_table.keys())[0]
    z_values = structured_defect_table[key0][_mixture_fraction_name]

    z_dim = Dimension(_mixture_fraction_name, z_values)
    x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, x_values, diss_rate_log_scaled)
    g_dim = Dimension(_enthalpy_defect_name + _stoich_suffix, g_values)

    output_library = Library(z_dim, x_dim, g_dim)
    output_library.extra_attributes['mech_spec'] = m

    for quantity in structured_defect_table[key0]:
        values = output_library.get_empty_dataset()

        for ix, x in enumerate(x_values):
            for ig, g in enumerate(g_values):
                values[:, ix, ig] = structured_defect_table[(x, g)][quantity]

                output_library[quantity] = values

    _write_library_footer(cput00, verbose)
    return output_library


def build_nonadiabatic_defect_transient_slfm_library(flamelet_specs,
                                                     diss_rate_values=np.logspace(-3, 2, 16),
                                                     diss_rate_ref='stoichiometric',
                                                     verbose=True,
                                                     solver_verbose=False,
                                                     h_stoich_spacing=10.e3,
                                                     num_procs=1,
                                                     integration_args=None,
                                                     n_defect_st=32,
                                                     extend_defect_dim=False,
                                                     diss_rate_log_scaled=True):
    """Build a flamelet library with the strained laminar flamelet model including heat loss effects through the enthalpy defect,
    where heat loss profiles are generated through rapid, transient extinction (as opposed to quasisteady heat loss)

    Parameters
    ----------
    flamelet_specs : dictionary or FlameletSpec instance
        data for the mechanism, streams, mixture fraction grid, etc.
    diss_rate_values : np.array
        reference dissipation rate values in the table (note that if the flamelet extinguishes at any point,
        the extinguished flamelet and larger dissipation rates are not included in the library)
    diss_rate_ref : str
        the reference point of the specified dissipation rate values, either 'stoichiometric' or 'maximum'
    verbose : bool
        whether or not to show progress of the library construction
    solver_verbose : bool
        whether or not to show detailed progress of sub-solvers in generating the library
    h_stoich_spacing : float
        the stoichiometric enthalpy spacing used in subsampling the transient solution history of each extinction solve
    n_defect_st : Int
        the number of stoichiometric enthalpy defect values to include in the library
    integration_args : kwargs
        extra arguments to be passed to the heat loss integration call (see Flamelet.integrate)
    num_procs : Int
        how many processors over which to distribute the parallel extinction solves
    extend_defect_dim : bool
        whether or not to add a buffer layer to the enthalpy defect field to aid in library lookups

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        the structured chemistry library

    """

    return _build_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                   'transient',
                                                   diss_rate_values,
                                                   diss_rate_ref,
                                                   verbose,
                                                   solver_verbose,
                                                   h_stoich_spacing,
                                                   num_procs,
                                                   integration_args,
                                                   n_defect_st,
                                                   extend_defect_dim,
                                                   diss_rate_log_scaled=diss_rate_log_scaled)


def build_nonadiabatic_defect_steady_slfm_library(flamelet_specs,
                                                  diss_rate_values=np.logspace(-3, 2, 16),
                                                  diss_rate_ref='stoichiometric',
                                                  verbose=True,
                                                  solver_verbose=False,
                                                  h_stoich_spacing=10.e3,
                                                  num_procs=1,
                                                  integration_args=None,
                                                  n_defect_st=32,
                                                  extend_defect_dim=False,
                                                  diss_rate_log_scaled=True):
    """Build a flamelet library with the strained laminar flamelet model including heat loss effects through the enthalpy defect,
    where heat loss profiles are generated through quasisteady extinction

    Parameters
    ----------
    flamelet_specs : dictionary or FlameletSpec instance
        data for the mechanism, streams, mixture fraction grid, etc.
    diss_rate_values : np.array
        reference dissipation rate values in the table (note that if the flamelet extinguishes at any point,
        the extinguished flamelet and larger dissipation rates are not included in the library)
    diss_rate_ref : str
        the reference point of the specified dissipation rate values, either 'stoichiometric' or 'maximum'
    verbose : bool
        whether or not to show progress of the library construction
    solver_verbose : bool
        whether or not to show detailed progress of sub-solvers in generating the library
    h_stoich_spacing : float
        the stoichiometric enthalpy spacing used in subsampling the transient solution history of each extinction solve
    n_defect_st : Int
        the number of stoichiometric enthalpy defect values to include in the library
    integration_args : kwargs
        extra arguments to be passed to the heat loss integration call (see Flamelet.integrate)
    num_procs : Int
        how many processors over which to distribute the parallel extinction solves
    extend_defect_dim : bool
        whether or not to add a buffer layer to the enthalpy defect field to aid in library lookups

    Returns
    -------
    library : spitfire.chemistry.library.Library instance
        the structured chemistry library

    """

    return _build_nonadiabatic_defect_slfm_library(flamelet_specs,
                                                   'steady',
                                                   diss_rate_values,
                                                   diss_rate_ref,
                                                   verbose,
                                                   solver_verbose,
                                                   h_stoich_spacing,
                                                   num_procs,
                                                   integration_args,
                                                   n_defect_st,
                                                   extend_defect_dim,
                                                   diss_rate_log_scaled=diss_rate_log_scaled)


try:
    from pytabprops import ClippedGaussMixMdl, BetaMixMdl, LagrangeInterpolant1D, StateTable


    def _pdf_object_and_info(pdf):
        if isinstance(pdf, str):
            is_custom_pdf = False
            if pdf.lower() == 'clipgauss':
                pdf_obj = ClipGaussPDF()
            elif pdf.lower() == 'beta':
                pdf_obj = BetaPDF()
            elif pdf.lower() == 'doubledelta':
                pdf_obj = DoubleDeltaPDF()
            else:
                raise ValueError("Unsupported PDF string")
        else:
            pdf_obj = pdf
            is_custom_pdf = not isinstance(pdf, ClipGaussPDF) and \
                            not isinstance(pdf, BetaPDF) and \
                            not isinstance(pdf, DoubleDeltaPDF)
        return is_custom_pdf, pdf_obj


    def _single_integral(is_custom_pdf, pdf_obj, indepvars, depvars, convolution_spline_order, integrator_intervals):
        if not is_custom_pdf:
            interp = LagrangeInterpolant1D(convolution_spline_order,
                                            indepvars, depvars, True)
            return pdf_obj.integrate(interp, integrator_intervals)
        else:
            interp = interp1d(indepvars, depvars, kind=convolution_spline_order,
                              fill_value=(float(depvars[0]), float(depvars[-1])),
                              bounds_error=False)
            return pdf_obj.integrate(interp)       
    

    def _convolve_full_property(p, managed_dict, turb_lib, lam_lib, pdf_spec):
        use_scaled_variance = pdf_spec.scaled_variance_values is not None
        variance_range = pdf_spec.scaled_variance_values if use_scaled_variance else pdf_spec.variance_values
        prop_turb = turb_lib[p]
        norm_fac = np.max(lam_lib[p]) - np.min(lam_lib[p]) if np.max(lam_lib[p]) - np.min(lam_lib[p])>0. else 1.
        prop_lam = lam_lib[p] / norm_fac
        is_custom_pdf, pdf_obj = _pdf_object_and_info(pdf_spec.pdf)

        for ((izm, zm), (isvm, svm)) in itertools.product(enumerate(turb_lib.dims[0].values),
                                                          enumerate(variance_range.copy())):
            pdf_obj.set_mean(zm)
            if use_scaled_variance:
                pdf_obj.set_scaled_variance(svm)
            else:
                pdf_obj.set_variance(svm)

            for indices in itertools.product(*[range(d.values.size) for d in turb_lib.dims[1:-1]]):
                lam_ex_list = [slice(idx, idx + 1) for idx in indices]
                lam_point_list = [slice(izm, izm + 1), *lam_ex_list]
                lam_line_list = [slice(None), *lam_ex_list]
                turb_slice = tuple(lam_point_list + [slice(isvm, isvm + 1)])
                prop_turb[turb_slice] = _single_integral(is_custom_pdf,
                                                         pdf_obj,
                                                         turb_lib.dims[0].values.ravel(),
                                                         prop_lam[tuple(lam_line_list)].ravel(),
                                                         pdf_spec.convolution_spline_order,
                                                         pdf_spec.integrator_intervals) * norm_fac
        if managed_dict is None:
            return prop_turb
        else:
            managed_dict[p] = prop_turb


    def _convolve_single_mean_variance(data, mean, variance, mean_range, pdf_spec):
        is_custom_pdf, pdf_obj = _pdf_object_and_info(pdf_spec.pdf)
        pdf_obj.set_mean(mean)
        pdf_obj.set_scaled_variance(variance) if pdf_spec.scaled_variance_values is not None else pdf_obj.set_variance(variance)
        
        if data.ndim == 1:
            return _single_integral(is_custom_pdf,
                                    pdf_obj,
                                    mean_range,
                                    data,
                                    pdf_spec.convolution_spline_order,
                                    pdf_spec.integrator_intervals)
        else:
            ans = np.zeros([*data.shape[1:]])
            for indices in itertools.product(*[range(data.shape[i+1]) for i in range(data.ndim-1)]):
                lam_ex_list = [slice(idx, idx + 1) for idx in indices]
                lam_line_list = [slice(None), *lam_ex_list]

                ans[tuple(lam_ex_list)] = _single_integral(is_custom_pdf,
                                                           pdf_obj,
                                                           mean_range,
                                                           data[tuple(lam_line_list)],
                                                           pdf_spec.convolution_spline_order,
                                                           pdf_spec.integrator_intervals)
            return ans


    def _emplace_single_mean_variance(output_dict, name, mean_index, var_index, norm_fac, convolve_output):
        ndims = len(output_dict.dims)
        if ndims>2:
            indices = [mean_index,*[slice(None) for i in range(ndims-2)], var_index]
            output_dict[name][tuple(indices)] = convolve_output * norm_fac
        else:
            output_dict[name][mean_index, var_index] = convolve_output * norm_fac


    def _convolve_single_mean(data, mean, variances, mean_range, pdf_spec):
        is_custom_pdf, pdf_obj = _pdf_object_and_info(pdf_spec.pdf)
        pdf_obj.set_mean(mean)
        toreturn = np.zeros_like(variances) if data.ndim == 1 else np.zeros([*data.shape[1:], variances.size])
        for iv,variance in enumerate(variances):
            pdf_obj.set_scaled_variance(variance) if pdf_spec.scaled_variance_values is not None else pdf_obj.set_variance(variance)
            
            if data.ndim == 1:
                toreturn[iv] = _single_integral(is_custom_pdf,
                                                pdf_obj,
                                                mean_range,
                                                data,
                                                pdf_spec.convolution_spline_order,
                                                pdf_spec.integrator_intervals)
            else:
                ans = np.zeros([*data.shape[1:]])
                for indices in itertools.product(*[range(data.shape[i+1]) for i in range(data.ndim-1)]):
                    lam_ex_list = [slice(idx, idx + 1) for idx in indices]
                    lam_line_list = [slice(None), *lam_ex_list]
                    ans[tuple(lam_ex_list)] = _single_integral(is_custom_pdf,
                                                               pdf_obj,
                                                               mean_range,
                                                               data[tuple(lam_line_list)],
                                                               pdf_spec.convolution_spline_order,
                                                               pdf_spec.integrator_intervals)
                toreturn[(*[slice(None) for i in range(data.ndim-1)],iv)] = ans
        return toreturn


    def _emplace_single_mean(output_dict, name, mean_index, norm_fac, convolve_output):
        ndims = len(output_dict.dims)
        if ndims>2:
            indices = [mean_index,*[slice(None) for i in range(ndims-1)]]
            output_dict[name][tuple(indices)] = convolve_output * norm_fac
        else:
            output_dict[name][mean_index, :] = convolve_output * norm_fac


    def _convolve_single_variance(data, means, variance, pdf_spec):
        is_custom_pdf, pdf_obj = _pdf_object_and_info(pdf_spec.pdf)
        toreturn = np.zeros_like(means) if data.ndim == 1 else np.zeros([means.size,*data.shape[1:]])
        for im,mean in enumerate(means):
            pdf_obj.set_mean(mean)
            pdf_obj.set_scaled_variance(variance) if pdf_spec.scaled_variance_values is not None else pdf_obj.set_variance(variance)
            
            if data.ndim == 1:
                toreturn[im] = _single_integral(is_custom_pdf,
                                                pdf_obj,
                                                means,
                                                data,
                                                pdf_spec.convolution_spline_order,
                                                pdf_spec.integrator_intervals)
            else:
                ans = np.zeros([*data.shape[1:]])
                for indices in itertools.product(*[range(data.shape[i+1]) for i in range(data.ndim-1)]):
                    lam_ex_list = [slice(idx, idx + 1) for idx in indices]
                    lam_line_list = [slice(None), *lam_ex_list]
                    ans[tuple(lam_ex_list)] = _single_integral(is_custom_pdf,
                                                               pdf_obj,
                                                               means,
                                                               data[tuple(lam_line_list)],
                                                               pdf_spec.convolution_spline_order,
                                                               pdf_spec.integrator_intervals)
                toreturn[(im,*[slice(None) for i in range(data.ndim-1)])] = ans
        return toreturn


    def _emplace_single_variance(output_dict, name, var_index, norm_fac, convolve_output):
        ndims = len(output_dict.dims)
        if ndims>2:
            indices = [*[slice(None) for i in range(ndims-1)], var_index]
            output_dict[name][tuple(indices)] = convolve_output * norm_fac
        else:
            output_dict[name][:, var_index] = convolve_output * norm_fac


    def _extend_presumed_pdf_first_dim(lam_lib, pdf_spec, added_suffix, num_procs, verbose=False):
        turb_dims = [Dimension(d.name + added_suffix, d.values, d.log_scaled) for d in lam_lib.dims]
        if pdf_spec.pdf != 'delta':
            turb_dims.append(Dimension(pdf_spec.variance_name,
                                       pdf_spec.scaled_variance_values if pdf_spec.scaled_variance_values is not None else pdf_spec.variance_values,
                                       pdf_spec.log_scaled))
            turb_lib = Library(*turb_dims)
            for p in lam_lib.props:
                turb_lib[p] = turb_lib.get_empty_dataset()

        if pdf_spec.pdf == 'delta':
            turb_lib = Library(*turb_dims)
            for p in lam_lib.props:
                turb_lib[p] = lam_lib[p].copy()
            return turb_lib

        if verbose:
            cput0 = perf_counter()
            num_integrals = len(turb_lib.props)
            for d in turb_lib.dims:
                num_integrals *= d.npts
            print(f'{pdf_spec.variance_name}: computing {num_integrals} integrals... ', end='', flush=True)

        if num_procs == 1:
            for p in turb_lib.props:
                turb_lib[p] = _convolve_full_property(p, None, turb_lib, lam_lib, pdf_spec)
        else:
            parallel_type = pdf_spec.parallel_type
            if parallel_type=='default':
                if isinstance(pdf_spec.pdf, str):
                    parallel_type = 'property-mean' if pdf_spec.pdf.lower() == 'beta' else 'property-variance'
                else:
                    parallel_type = 'property-mean' if isinstance(pdf_spec.pdf, BetaPDF) else 'property-variance'
            
            if parallel_type=='full':
                variance_range = pdf_spec.scaled_variance_values if pdf_spec.scaled_variance_values is not None else pdf_spec.variance_values
                mean_range = turb_lib.dims[0].values
                with Pool(processes=num_procs) as pool:
                    tasks = set()
                    for name in lam_lib.props:
                        data = lam_lib[name]
                        norm_fac = np.max(data) - np.min(data) if np.max(data) - np.min(data)>0. else 1.
                        data /= norm_fac
                        for mean_index, mean in enumerate(mean_range):
                            for var_index, variance in enumerate(variance_range):
                                tasks.add(pool.apply_async(partial(_convolve_single_mean_variance, data, mean, variance, mean_range, pdf_spec), 
                                                           callback=partial(_emplace_single_mean_variance, turb_lib, name, mean_index, var_index, norm_fac)))
                    for t in tasks:
                        t.get()
            elif parallel_type=='property-variance':
                variance_range = pdf_spec.scaled_variance_values if pdf_spec.scaled_variance_values is not None else pdf_spec.variance_values
                mean_range = turb_lib.dims[0].values
                with Pool(processes=num_procs) as pool:
                    tasks = set()
                    for name in lam_lib.props:
                        data = lam_lib[name]
                        norm_fac = np.max(data) - np.min(data) if np.max(data) - np.min(data)>0. else 1.
                        data /= norm_fac
                        for var_index, variance in enumerate(variance_range):
                            tasks.add(pool.apply_async(partial(_convolve_single_variance, data, mean_range, variance, pdf_spec), 
                                                       callback=partial(_emplace_single_variance, turb_lib, name, var_index, norm_fac)))
                    for t in tasks:
                        t.get()
            elif parallel_type=='property-mean':
                variance_range = pdf_spec.scaled_variance_values if pdf_spec.scaled_variance_values is not None else pdf_spec.variance_values
                mean_range = turb_lib.dims[0].values
                with Pool(processes=num_procs) as pool:
                    tasks = set()
                    for name in lam_lib.props:
                        data = lam_lib[name]
                        norm_fac = np.max(data) - np.min(data) if np.max(data) - np.min(data)>0. else 1.
                        data /= norm_fac
                        for mean_index, mean in enumerate(mean_range):
                            tasks.add(pool.apply_async(partial(_convolve_single_mean, data, mean, variance_range, mean_range, pdf_spec), 
                                                       callback=partial(_emplace_single_mean, turb_lib, name, mean_index, norm_fac)))
                    for t in tasks:
                        t.get()
            elif parallel_type=='property':
                pool = Pool(processes=num_procs)
                manager = Manager()
                managed_dict = manager.dict()
                pool.map(partial(_convolve_full_property,
                                managed_dict=managed_dict,
                                turb_lib=turb_lib,
                                lam_lib=lam_lib,
                                pdf_spec=pdf_spec),
                        turb_lib.props)
                for p in managed_dict:
                    turb_lib[p] = managed_dict[p]
                pool.close()
                pool.join()
            else:
                raise ValueError("Unsupported parallel_type.")

        if verbose:
            cputf = perf_counter()
            dcput = cputf - cput0
            avg = float(num_integrals) / dcput
            print(f'completed in {dcput:.1f} seconds, average = {int(avg)} integrals/s.')
        return turb_lib


    class ClipGaussPDF:
        """Clipped Gaussian presumed PDF wrapper for the Tabprops implementation.

        Parameters
        ----------
        nfpts : Int
            number of modified means to tabulate over, default 201
        ngpts : Int
            number of modified variances to tabulate over, default 201
        write_params_to_disk : bool
            whether or not to write out the tabulated modified means and variances, default False
        """
        def __init__(self, 
                     nfpts=201, 
                     ngpts=201,
                     write_params_to_disk=False):
            self._pdf = ClippedGaussMixMdl(nfpts, ngpts, write_params_to_disk)
            self._nfpts = nfpts
            self._ngpts = ngpts
            self._write_params_to_disk = write_params_to_disk

        def __getstate__(self):
            params = {}
            params['nfpts'] = self._nfpts
            params['ngpts'] = self._ngpts
            params['write_params_to_disk'] = self._write_params_to_disk
            return params

        def __setstate__(self, state):
            self.__init__(**state)
        
        def get_pdf(self, x):
            "Evaluate the clipped Gaussian PDF"
            return self._pdf.get_pdf(x)
        
        def set_mean(self, mean):
            "Set the mean for the clipped Gaussian PDF"
            self._pdf.set_mean(mean)

        def set_variance(self, variance):
            "Set the variance for the clipped Gaussian PDF"
            self._pdf.set_variance(variance)

        def set_scaled_variance(self, variance):
            "Set the scaled variance for the clipped Gaussian PDF"
            self._pdf.set_scaled_variance(variance)

        def get_mean(self):
            "returns the mean for the clipped Gaussian PDF"
            return self._pdf.get_mean()

        def get_variance(self):
            "returns the variance for the clipped Gaussian PDF"
            return self._pdf.get_variance()

        def get_scaled_variance(self):
            "returns the scaled variance for the clipped Gaussian PDF"
            return self._pdf.get_scaled_variance()
        
        def get_max_variance(self):
            "returns the maximum variance for the clipped Gaussian PDF"
            mean = self.get_mean()
            return mean * (1.-mean)
        
        def integrate(self, interpolant_obj, integrator_intervals=100):
            """Perform the convolution of the provided interpolant with the clipped Gaussian PDF.

            Parameters
            ----------
            interpolant_obj : Tabprops Lagrange interpolant
                Interpolant for evaluating the property in the convolution integrals.
            integrator_intervals : Int
                number of subintervals for integration, default 100
            """
            return self._pdf.integrate(interpolant_obj, integrator_intervals)


    class BetaPDF:
        """Beta presumed PDF leverages the Tabprops implementation with improved integration.

        Parameters
        ----------
        scaled_variance_max_integrate : float
            The maximum scalar variance for which integration is trusted. Any values higher than 
            this will have the integral interpolated between the integration at scaled_variance_max_integrate and 1.
            Default value is 0.86.
        scaled_variance_min_integrate : float
            The minimum scalar variance for which integration is trusted. Any values lower than 
            this will have the integral interpolated between the integration at scaled_variance_min_integrate and 0.
            Default value is 6e-4.
        mean_boundary_integrate : float
            The distance on either side of the [0,1] mean boundary for which integration is trusted. Any values within 
            a shorter distance from the boundary will have the integral interpolated between the integration at 
            the trusted distance and the nearest boundary.
            Default value is 6e-5.
        """
        def __init__(self, 
                     scaled_variance_max_integrate=0.86, 
                     scaled_variance_min_integrate=6.e-4,
                     mean_boundary_integrate=6.e-5):
            self._oldbeta = BetaMixMdl() # Beta PDF in Tabprops
            self._min_bound = 0. # lower bound for integral
            self._max_bound = 1. # upper bound for integral
            self._scaled_variance_max_integrate = scaled_variance_max_integrate
            self._scaled_variance_min_integrate = scaled_variance_min_integrate
            self._mean_boundary_integrate = mean_boundary_integrate

        def __getstate__(self):
            params = {}
            params['scaled_variance_max_integrate'] = self._scaled_variance_max_integrate
            params['scaled_variance_min_integrate'] = self._scaled_variance_min_integrate
            params['mean_boundary_integrate'] = self._mean_boundary_integrate
            return params

        def __setstate__(self, state):
            self.__init__(**state)

        def get_pdf(self, x):
            "Evaluate the Beta PDF"
            return self._oldbeta.get_pdf(x)

        def set_mean(self, mean):
            "Set the mean for the Beta PDF"
            self._oldbeta.set_mean(mean)

        def set_variance(self, variance):
            "Set the variance for the Beta PDF"
            self._oldbeta.set_variance(variance)

        def set_scaled_variance(self, variance):
            "Set the scaled variance for the Beta PDF"
            self._oldbeta.set_scaled_variance(variance)

        def get_mean(self):
            "returns the mean for the Beta PDF"
            return self._oldbeta.get_mean()

        def get_variance(self):
            "returns the variance for the Beta PDF"
            return self._oldbeta.get_variance()

        def get_scaled_variance(self):
            "returns the scaled variance for the Beta PDF"
            return self._oldbeta.get_scaled_variance()
        
        def get_max_variance(self):
            "returns the maximum variance for the Beta PDF"
            mean = self.get_mean()
            return mean * (1.-mean)
        
        def _compute_unmixed_state(self, interpolant):
            "the convolution at a scaled variance of 1"
            mean = self.get_mean()
            t0 = interpolant(0.)
            t1 = interpolant(1.)
            return t1 * mean + t0*(1.-mean)
        
        def _compute_mixed_state(self, interpolant):
            "the convolution at a scaled variance of 0"
            return interpolant(self.get_mean())

        def _interp_variance(self, integral_dict, scaled_variance_copy, bound):
            "convolution is interpolated between trusted variance and boundary"
            if bound=='max':
                bound_integral = self._compute_unmixed_state(integral_dict['interpolant'])
                bound_var = 1.
                integrate_var = self._scaled_variance_max_integrate
            else:
                bound_integral = self._compute_mixed_state(integral_dict['interpolant'])
                bound_var = 0.
                integrate_var = self._scaled_variance_min_integrate
            self.set_scaled_variance(integrate_var)
            cutoff = integrate.quad(integral_dict['integrand'], self._min_bound, self._max_bound, limit=integral_dict['integrator_intervals'], points=integral_dict['points'], full_output=1, epsabs=integral_dict['epsabs'], epsrel=integral_dict['epsrel'])[0]
            self.set_scaled_variance(scaled_variance_copy)
            return cutoff + (bound_integral - cutoff) / (bound_var - integrate_var) * (scaled_variance_copy - integrate_var)

        def _interp_mean(self, integral_dict, mean_copy, withvariance=False, variance_bound=None):
            "convolution is interpolated between trusted mean and boundary"
            scaled_variance_copy = np.copy(self.get_scaled_variance())
            if mean_copy<self._mean_boundary_integrate:
                bound_var = 0.
                integrate_var=self._mean_boundary_integrate
            else:
                bound_var = 1.
                integrate_var=1.-self._mean_boundary_integrate
            bound_integral = integral_dict['interpolant'](bound_var)
            self.set_mean(integrate_var)
            self.set_scaled_variance(scaled_variance_copy)
            if withvariance:
                cutoff = self._interp_variance(integral_dict, scaled_variance_copy, variance_bound)
            else:
                cutoff = integrate.quad(integral_dict['integrand'], self._min_bound, self._max_bound, limit=integral_dict['integrator_intervals'], points=integral_dict['points'], full_output=1, epsabs=integral_dict['epsabs'], epsrel=integral_dict['epsrel'])[0]
            self.set_mean(mean_copy)
            self.set_scaled_variance(scaled_variance_copy)
            return cutoff + (bound_integral - cutoff) / (bound_var - integrate_var) * (mean_copy - integrate_var)

        def integrate(self, interpolant_obj, integrator_intervals=100):
            """Perform the convolution of the provided interpolant with the Beta PDF.

            Parameters
            ----------
            interpolant_obj : Tabprops Lagrange interpolant
                Interpolant for evaluating the property in the convolution integrals.
            integrator_intervals : Int
                upper bound on the number of subintervals used in the adaptive algorithm, default 100
            """
            tbl = StateTable()
            tbl.add_entry("val", interpolant_obj, ['x'])
            interpolant = lambda x: tbl.query("val", x)

            scaled_variance = self.get_scaled_variance()
            mean = self.get_mean()

            zerotol = 1.e-12
            if mean<zerotol or mean>=1.-zerotol:
                return interpolant(mean)
            
            if self.get_variance() < 1.e-8:  # use mean values
                return self._compute_mixed_state(interpolant)
            elif scaled_variance > 0.99: # use unmixed values
                return self._compute_unmixed_state(interpolant)
            else:
                epsabs = 1.e-10 if scaled_variance<1.e-3 else 1.49e-8
                epsrel=epsabs

                def integrand(x):
                    return interpolant(x) * self.get_pdf(x)

                integral_dict = {}
                integral_dict['integrand'] = integrand
                integral_dict['interpolant'] = interpolant
                integral_dict['integrator_intervals'] = integrator_intervals
                integral_dict['epsabs'] = epsabs
                integral_dict['epsrel'] = epsrel
                integral_dict['points'] = [0.,1.]
                
                if scaled_variance > self._scaled_variance_max_integrate: # interpolate variance
                    if mean<self._mean_boundary_integrate or mean>1.-self._mean_boundary_integrate: # interpolate mean
                        return self._interp_mean(integral_dict, mean, True, 'max')
                    else:
                        return self._interp_variance(integral_dict, scaled_variance,'max')

                elif scaled_variance < self._scaled_variance_min_integrate: # interpolate variance
                    if mean<self._mean_boundary_integrate or mean>1.-self._mean_boundary_integrate: # interpolate mean
                        return self._interp_mean(integral_dict, mean, True, 'min')
                    else:
                        return self._interp_variance(integral_dict, scaled_variance,'min')
                else:
                    if mean<self._mean_boundary_integrate or mean>1.-self._mean_boundary_integrate: # interpolate mean
                        return self._interp_mean(integral_dict, mean, False)
                    else: # integrate
                        return integrate.quad(integrand, self._min_bound, self._max_bound, limit=integral_dict['integrator_intervals'], points=integral_dict['points'], full_output=1, epsabs=integral_dict['epsabs'], epsrel=integral_dict['epsrel'])[0]


    class DoubleDeltaPDF:
        """Double Delta PDF."""
        def __init__(self):
            self._mean = 0.
            self._variance = 0.
            self._scaled_variance = 0.
            self._max_variance = 0.

        def set_mean(self, mean):
            "set the mean for the Double Delta PDF"
            self._mean = mean
            self._varmax = mean * (1. - mean)

        def set_variance(self, variance):
            "set the variance for the Double Delta PDF"
            self._variance = variance
            if self._varmax > 0:
                self._scaled_variance = variance / self._varmax
            else:
                self._scaled_variance = 0.

        def set_scaled_variance(self, scaled_variance):
            "set the scaled variance for the Double Delta PDF"
            self._scaled_variance = scaled_variance
            self._variance = scaled_variance * self._varmax

        def get_mean(self):
            "returns the mean of the Double Delta PDF"
            return self._mean

        def get_variance(self):
            "returns the variance of the Double Delta PDF"
            return self._variance

        def get_scaled_variance(self):
            "returns the scaled variance of the Double Delta PDF"
            return self._scaled_variance

        def find_bounds(self):
            "returns the left and right delta location along with the left delta weight"
            left = self._mean - np.sqrt(self._variance)
            right = self._mean + np.sqrt(self._variance)
            weight_left = 0.5
            if left < 0.:
                left = 0.
                weight_left = 1. - self._mean**2/(self._variance + self._mean**2)
                right = self._mean / (1.-weight_left)
            elif right > 1.:
                right = 1.
                weight_left = (1.-self._mean)**2 / (self._variance + (1.-self._mean)**2)
                left = 1. - (1.-self._mean)/weight_left
            if left < 0. or right > 1.:
                raise ValueError(f"cannot support (mean, scaled variance) of ({self._mean:.2f}, {self._scaled_variance:.2e}) on domain [0,1].")
            return left,right,weight_left

        def get_pdf(self, x):
            "evaluate the Double Delta PDF integral"
            left,right,weight_left = self.find_bounds()
            pdf = np.zeros_like(x)
            pdf[x==left] = weight_left
            pdf[x==right] = 1.-weight_left
            return pdf
        
        def _compute_unmixed_state(self, interpolant):
            "the convolution at a scaled variance of 1"
            mean = self.get_mean()
            t0 = interpolant(0.)
            t1 = interpolant(1.)
            return t1 * mean + t0*(1.-mean)
        
        def _compute_mixed_state(self, interpolant):
            "the convolution at a scaled variance of 0"
            return interpolant(self.get_mean())
        
        def integrate(self, interpolant_obj, integrator_intervals=100):
            """Perform the convolution of the provided interpolant with the Double Delta PDF.

            Parameters
            ----------
            interpolant_obj : Tabprops Lagrange interpolant
                Interpolant for evaluating the property in the convolution integrals.
            integrator_intervals : Int
                This variable is not used for Double Delta integration
            """
            tbl = StateTable()
            tbl.add_entry("val", interpolant_obj, ['x'])
            interpolant = lambda x: tbl.query("val", x)
            if self.get_variance() < 1.e-8:  # use mean values
                return self._compute_mixed_state(interpolant)
            elif self.get_scaled_variance() > 0.99: # use unmixed values
                return self._compute_unmixed_state(interpolant)
            else:
                left,right,weight_left = self.find_bounds()
                return weight_left*interpolant(left) + (1.-weight_left)*interpolant(right)


    def compute_pdf_max_integration_errors(pdf, means, scaled_variances, relative_tolerance=1.e-6, integrator_intervals=100):
        """Compute the maximum relative error magnitude for the provided PDF satisfying the following 3 integrals over a range of means and scaled variances
            
            1. :math:`1 = \\int_{-\infty}^\infty P(\\phi) \\mathrm{d}\\phi`

            2. :math:`\\bar{\\phi} = \\int_{-\infty}^\infty \\phi P(\\phi) \\mathrm{d}\\phi`

            3. :math:`\\sigma_{\\phi}^2 = \\int_{-\infty}^\infty (\\phi - \\bar{\\phi})^2 P(\\phi) \\mathrm{d}\\phi`

            Parameters
            ----------
            pdf : a constructed pdf class
                the pdf for which to evaluate the integral errors
            means : numpy array or list
                the set of means at which to evaluate the integral errors
            scaled_variances : numpy array or list
                the set of scaled variances at which to evaluate the integral errors
            relative_tolerance : float
                the offset used in the denominator for computing relative errors
            integrator_intervals : float
                subintervals for integration when applicable

            Returns
            -------
            maximum relative error for satisfying integral 1, maximum relative error for satisfying integral 2, maximum relative error for satisfying integral 3
        """
        errors = np.zeros((scaled_variances.size * means.size))
        linspace = means.copy()
        interp = LagrangeInterpolant1D(3, linspace, np.ones_like(linspace), False)
        i = 0
        for svar in scaled_variances:
            for mean in means:
                pdf.set_mean(mean)
                pdf.set_scaled_variance(svar)
                integral = pdf.integrate(interp, integrator_intervals)
                errors[i] = integral - 1.
                i += 1
        maxerr_pdf = np.max(np.abs(errors))

        interp = LagrangeInterpolant1D(3, linspace, linspace, False)
        i = 0
        for svar in scaled_variances:
            for mean in means:
                pdf.set_mean(mean)
                pdf.set_scaled_variance(svar)
                integral = pdf.integrate(interp, integrator_intervals)
                errors[i] = (integral - mean)/(mean+relative_tolerance)
                i += 1
        maxerr_mean = np.max(np.abs(errors))
    
        i = 0
        for svar in scaled_variances:
            for mean in means:
                pdf.set_mean(mean)
                pdf.set_scaled_variance(svar)
                maxvar = mean * (1.-mean)
                interp = LagrangeInterpolant1D(3, linspace, (linspace-mean)**2, False)
                integral = pdf.integrate(interp, integrator_intervals)
                errors[i] = (integral - svar*maxvar)/(svar*maxvar+relative_tolerance)
                i += 1
        maxerr_var = np.max(np.abs(errors))

        return maxerr_pdf, maxerr_mean, maxerr_var


except ImportError:
    pass


def require_pytabprops(method_name):
    try:
        from pytabprops import ClippedGaussMixMdl, BetaMixMdl, LagrangeInterpolant1D, StateTable
    except ImportError:
        raise ModuleNotFoundError(f'{method_name} requires the pytabprops package')


def _apply_presumed_pdf_1var(library, variable_name, pdf_spec, added_suffix=_mean_suffix, num_procs=1, verbose=False):
    require_pytabprops('apply_presumed_PDF_model')

    index = 0
    for d in library.dims:
        if d.name == variable_name:
            break
        else:
            index += 1
    if index == len(library.dims):
        raise KeyError(f'Invalid variable name \"{variable_name}\" for library with names {library.dim_names}')

    _pdf_spec = copy.copy(pdf_spec)
    if _pdf_spec.variance_name is None:
        if variable_name == _mixture_fraction_name:
            _pdf_spec.variance_name = _scaled_scalar_variance_name
        else:
            _pdf_spec.variance_name = variable_name + '_variance'

    if index == 0:
        library_t = _extend_presumed_pdf_first_dim(library, _pdf_spec, added_suffix, num_procs, verbose)
    else:
        swapped_dims = library.dims
        swapped_dims[index], swapped_dims[0] = swapped_dims[0], swapped_dims[index]
        swapped_lib = Library(*swapped_dims)
        for p in library.props:
            swapped_lib[p] = np.swapaxes(library[p], 0, index)
        swapped_lib_t = _extend_presumed_pdf_first_dim(swapped_lib, _pdf_spec, added_suffix, num_procs, verbose)
        swapped_lib_t_dims = copy.copy(swapped_lib_t.dims)
        swapped_lib_t_dims[0], swapped_lib_t_dims[index] = swapped_lib_t_dims[index], swapped_lib_t_dims[0]
        library_t = Library(*swapped_lib_t_dims)
        for p in swapped_lib_t.props:
            library_t[p] = np.swapaxes(swapped_lib_t[p], index, 0)

    for key in library.extra_attributes:
        library_t.extra_attributes[key] = copy.copy(library.extra_attributes[key])

    return library_t


class PDFSpec:
    def __init__(self,
                 pdf='delta',
                 scaled_variance_values=None,
                 variance_values=None,
                 convolution_spline_order=3,
                 integrator_intervals=100,
                 variance_name=None,
                 log_scaled=False,
                 parallel_type='default'):
        """Specification of a presumed PDF and integrator/spline details for a given single dimension in a library.

        Parameters
        ----------
        pdf : str, or custom object type
            the PDF, either 'ClipGauss' or 'Beta' for TabProps methods, 'DoubleDelta', or any custom object that implements the
            set_mean(), set_variance() or set_scaled_variance(), and integrate(scipy.interpolate.interp1d) methods.
        scaled_variance_values : np.array
            array of values of the scaled variance (varies between zero and one), provide this or variance_values
        variance_values : np.array
            array of values of the variance, provide this or scaled_variance_values
        variance_name : str
            the name of the variance dimension added to the output library
        convolution_spline_order : Int
            the order of the 1-D piecewise Lagrange reconstruction used in the convolution integrals, default is 3 (cubic)
        integrator_intervals : Int
            extra parameter provided to TabProps integrators, default 100
        variance_name : str
            the name of the variance dimension to be added, by default Spitfire will add "_variance" to the name
            of the dimension being convolved, or use "scaled_scalar_variance_mean" for "mixture_fraction"
        log_scaled : bool
            whether or not the dimension is populated with log-scaled values, default False
        parallel_type : str
            parallelization options include 'property' for properties alone, 'property-mean' for means and properties,
            'property-variance' for variances and properties, 'full' for means, variances, and properties,
            or 'default' which estimates the fastest option based on the pdf type
        """
        require_pytabprops('PDFSpec')

        self.pdf = pdf
        self.scaled_variance_values = scaled_variance_values
        self.variance_values = variance_values
        self.convolution_spline_order = convolution_spline_order
        self.integrator_intervals = integrator_intervals
        self.variance_name = variance_name
        self.log_scaled = log_scaled

        supported_parallel_types = ['full', 'property', 'property-variance', 'property-mean', 'default']
        if parallel_type not in supported_parallel_types:
            raise ValueError(f"Unsupported parallel_type '{parallel_type}'. Options include: {supported_parallel_types}")
        else:
            self.parallel_type = parallel_type


def apply_mixing_model(library, mixing_spec, added_suffix=_mean_suffix, num_procs=1, verbose=False):
    """Take an existing tabulated chemistry library and incorporate subgrid variation in each reaction variable with a
        presumed PDF model. This requires statistical independence of the reaction variables. If a reaction variable
        is not included in the mixing_spec dictionary, a delta PDF is presumed for it.

    Parameters
    ----------
    library : spitfire.Library
        an existing library from a reaction model
    mixing_spec : str
        a dictionary mapping reaction variable names to PDFSpec objects that describe the variable's presumed PDF
    added_suffix : str
        string to add to each name, for instance '_mean' if this is the first PDF convolution, or '' if a successive convolution
    num_procs : Int
        how many processors over which to distribute the parallel extinction solves
    verbose : bool
        whether or not (default False) to print information about the PDF convolution

    Returns
    -------
    library : spitfire.Library instance
        the library with any nontrivial variance dimensions added
    """
    require_pytabprops('apply_mixing_model')

    for dim in mixing_spec:
        if dim not in library.dim_names:
            raise KeyError(f'Invalid variable name \"{dim}\" for library with names {library.dim_names}')

    mixing_spec = copy.copy(mixing_spec)
    for dim in library.dims:
        if dim.name not in mixing_spec:
            mixing_spec[dim.name] = 'delta'

    for dim in mixing_spec:
        if mixing_spec[dim] == 'delta':
            mixing_spec[dim] = PDFSpec('delta', variance_values=np.array([0.]))

    for dim in mixing_spec:
        if (mixing_spec[dim].variance_values is None and
                mixing_spec[dim].scaled_variance_values is None and
                mixing_spec[dim].pdf == 'delta'):
            mixing_spec[dim] = PDFSpec('delta', variance_values=np.array([0.]))

    extensions = []
    for i, key in enumerate(mixing_spec):
        extensions.append({'variable_name': key,
                           'pdf_spec': mixing_spec[key],
                           'num_procs': num_procs,
                           'verbose': verbose})

    def get_size(element):
        pdfspec = element['pdf_spec']
        if pdfspec.pdf == 'delta':
            return 0
        else:
            if pdfspec.variance_values is not None:
                return pdfspec.variance_values.size
            else:
                return pdfspec.scaled_variance_values.size

    extensions = sorted(extensions, key=lambda element: get_size(element))
    turb_lib = Library.copy(library)
    for i, ext in enumerate(extensions):
        ext['library'] = turb_lib
        ext['added_suffix'] = '' if i < len(extensions) - 1 else added_suffix
        turb_lib = _apply_presumed_pdf_1var(**ext)
        if get_size(ext) == 1 and ext['pdf_spec'].pdf != 'delta':
            turb_lib = Library.squeeze(turb_lib[tuple([slice(None)] * (len(turb_lib.dims) - 1) + [slice(0, 1)])])

    if 'mixing_spec' in turb_lib.extra_attributes:
        turb_lib.extra_attributes['mixing_spec'].update(mixing_spec)
    else:
        turb_lib.extra_attributes['mixing_spec'] = mixing_spec
    return turb_lib
