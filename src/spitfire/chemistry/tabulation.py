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
        print(f'- mechanism: {mech.mech_xml_path}')
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
                                 include_extinguished=False):
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
        x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, np.array(x_values))

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
    flamelet_specs.convection_coefficient = 1.e7
    flamelet_specs.radiative_emissivity = 0.

    integration_args = dict(
        {'first_time_step': 1.e-9,
         'max_time_step': 1.e-1,
         'write_log': solver_verbose,
         'log_rate': 100})

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
                                            extend_defect_dim=False):
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
    x_dim = Dimension(_dissipation_rate_name + _stoich_suffix, x_values)
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
                                                     extend_defect_dim=False):
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
                                                   extend_defect_dim)


def build_nonadiabatic_defect_steady_slfm_library(flamelet_specs,
                                                  diss_rate_values=np.logspace(-3, 2, 16),
                                                  diss_rate_ref='stoichiometric',
                                                  verbose=True,
                                                  solver_verbose=False,
                                                  h_stoich_spacing=10.e3,
                                                  num_procs=1,
                                                  integration_args=None,
                                                  n_defect_st=32,
                                                  extend_defect_dim=False):
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
                                                   extend_defect_dim)


try:
    from pytabprops import ClippedGaussMixMdl, BetaMixMdl, LagrangeInterpolant1D


    class PDFSpec:
        def __init__(self,
                     pdf='delta',
                     scaled_variance_values=None,
                     variance_values=None,
                     convolution_spline_order=3,
                     integrator_intervals=1,
                     variance_name=None):
            """Specification of a presumed PDF and integrator/spline details for a given single dimension in a library.

            Parameters
            ----------
            pdf : str, or custom object type
                the PDF, either 'ClipGauss' or 'Beta' for TabProps methods, or any custom object that implements the
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
                extra parameter provided to TabProps integrators, default 1
            variance_name : str
                the name of the variance dimension to be added, by default Spitfire will add "_variance" to the name
                of the dimension being convolved, or use "scaled_scalar_variance_mean" for "mixture_fraction"
            """
            self.pdf = pdf
            self.scaled_variance_values = scaled_variance_values
            self.variance_values = variance_values
            self.convolution_spline_order = convolution_spline_order
            self.integrator_intervals = integrator_intervals
            self.variance_name = variance_name


    def _convolve_full_property(p, managed_dict, turb_lib, lam_lib, pdf_spec):
        use_scaled_variance = pdf_spec.scaled_variance_values is not None
        variance_range = pdf_spec.scaled_variance_values if use_scaled_variance else pdf_spec.variance_values
        prop_turb = turb_lib[p]
        prop_lam = lam_lib[p]
        if isinstance(pdf_spec.pdf, str):
            use_pytabprops = True
            pdf_obj = ClippedGaussMixMdl(201, 201, False) if pdf_spec.pdf == 'ClipGauss' else BetaMixMdl()
        else:
            pdf_obj = pdf_spec.pdf
            use_pytabprops = isinstance(pdf_spec.pdf, ClippedGaussMixMdl) or isinstance(pdf_spec.pdf, BetaMixMdl)

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
                if use_pytabprops:
                    interp = LagrangeInterpolant1D(pdf_spec.convolution_spline_order,
                                                   turb_lib.dims[0].values,
                                                   prop_lam[tuple(lam_line_list)], True)
                    prop_turb[turb_slice] = pdf_obj.integrate(interp, pdf_spec.integrator_intervals)
                else:
                    interp = interp1d(turb_lib.dims[0].values.ravel(), prop_lam[tuple(lam_line_list)].ravel(),
                                      kind=pdf_spec.convolution_spline_order,
                                      fill_value=(float(prop_lam[tuple(lam_line_list)][0]),
                                                  float(prop_lam[tuple(lam_line_list)][-1])),
                                      bounds_error=False)
                    prop_turb[turb_slice] = pdf_obj.integrate(interp)
        if managed_dict is None:
            return prop_turb
        else:
            managed_dict[p] = prop_turb


    def _extend_presumed_pdf_first_dim(lam_lib, pdf_spec, added_suffix, num_procs, verbose=False):
        turb_dims = [Dimension(d.name + added_suffix, d.values) for d in lam_lib.dims]
        if pdf_spec.pdf != 'delta':
            turb_dims.append(Dimension(pdf_spec.variance_name,
                                       pdf_spec.scaled_variance_values if pdf_spec.scaled_variance_values is not None else pdf_spec.variance_values))
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

        if verbose:
            cputf = perf_counter()
            dcput = cputf - cput0
            avg = float(num_integrals) / dcput
            print(f'completed in {dcput:.1f} seconds, average = {int(avg)} integrals/s.')
        return turb_lib
except ImportError:
    pass


def require_pytabprops(method_name):
    try:
        from pytabprops import ClippedGaussMixMdl, BetaMixMdl, LagrangeInterpolant1D
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
