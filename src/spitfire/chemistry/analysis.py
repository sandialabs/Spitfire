"""
This module contains thermochemical analysis tools such as property evaluators and chemical explosive mode analysis,
which operate generally on spitfire Library instances from time integration, flamelet tabulation, etc.
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
#
# You should have received a copy of the 3-clause BSD License
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#
# Questions? Contact Mike Hansen (mahanse@sandia.gov)


import numpy as np
import cantera as ct
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.library import Library
from scipy.linalg import eig, eigvals
from spitfire.chemistry.ctversion import check as cantera_version_check



def get_ct_solution_array(mechanism=None, library=None):
    """Obtain a Cantera SolutionArray object representative of an arbitrary library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        solution_array : cantera.SolutionArray instance
            a SolutionArray filled with the library data
        library_shape : tuple
            shape of the original library
        """

    if not isinstance(library, Library):
        raise TypeError('Input argument "library" to get_ct_solution_array() must be a Library')

    if mechanism is None:
        if 'mech_spec' in library.extra_attributes:
            mechanism = library.extra_attributes['mech_spec']
        else:
            raise ValueError('Input argument "mechanism" to get_ct_solution_array() is required as a mechanism could not be found at library.extra_attributes["mech_spec"]')

    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Invalid "mechanism" for get_ct_solution_array(): must be a ChemicalMechanismSpec')

    required_strings = ['temperature'] + ['mass fraction ' + s for s in mechanism.species_names]
    for required_string in required_strings:
        if required_string not in library:
            raise ValueError('Library argument to get_ct_solution_array() must contain '
                             'temperature, all mass fractions, and pressure or density')
    if 'density' not in library and 'pressure' not in library:
        raise ValueError('Library argument to get_ct_solution_array() must contain '
                         'temperature, all mass fractions, and pressure or density')

    library_shape = library['temperature'].shape
    nstates = library['temperature'].size
    Y = np.ndarray((nstates, mechanism.n_species))
    for i, name in enumerate(mechanism.species_names):
        Y[:, i] = library['mass fraction ' + name].ravel()
    states = ct.SolutionArray(mechanism.gas, (nstates,))

    if 'pressure' in library:
        states.TPY = library['temperature'].ravel(), library['pressure'].ravel(), Y
    elif 'density' in library:
        states.TDY = library['temperature'].ravel(), library['density'].ravel(), Y
    return states, library_shape


def compute_specific_enthalpy(mechanism, output_library):
    """Add the total specific enthalpy to a library (Cantera "enthalpy_mass"), named 'enthalpy'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'enthalpy' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['enthalpy'] = ctsol.enthalpy_mass.reshape(lib_shape)
    return output_library


def compute_density(mechanism, output_library):
    """Add the total mass density to a library (Cantera "density"), named 'density'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'density' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['density'] = ctsol.density.reshape(lib_shape)
    return output_library


def compute_pressure(mechanism, output_library):
    """Add the pressure to a library (Cantera "P"), named 'pressure'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'pressure' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['pressure'] = ctsol.P.reshape(lib_shape)
    return output_library


def compute_viscosity(mechanism, output_library):
    """Add the dynamic viscosity to a library (Cantera "viscosity"), named 'viscosity'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'viscosity' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['viscosity'] = ctsol.viscosity.reshape(lib_shape)
    return output_library


def compute_isobaric_specific_heat(mechanism, output_library):
    """Add the constant-pressure specific heat capacity to a library (Cantera "cp_mass"), named 'heat capacity cp'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'heat capacity cp' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['heat capacity cp'] = ctsol.cp_mass.reshape(lib_shape)
    return output_library


def compute_isochoric_specific_heat(mechanism, output_library):
    """Add the constant-volume specific heat capacity to a library (Cantera "cv_mass"), named 'heat capacity cv'

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)

        Returns
        -------
        output_library : cantera.SolutionArray instance
            the library with the 'heat capacity cv' field added
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
    output_library['heat capacity cv'] = ctsol.cv_mass.reshape(lib_shape)
    return output_library


def explosive_mode_analysis(mechanism,
                            output_library,
                            configuration='isobaric',
                            heat_transfer='adiabatic',
                            compute_explosion_indices=False,
                            compute_participation_indices=False,
                            include_secondary_mode=False):
    """Perform chemical explosive mode analysis across a range of states in a library.

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        configuration: str
            whether to analyze the system as "isobaric" (default) or "isochoric"
        heat_transfer: str
            whether to analyze the system as "adiabatic" (default) or "isothermal"
        compute_explosion_indices: bool
            whether or not (default: False) to include explosion index analysis
        compute_participation_indices: bool
            whether or not (default: False) to include participation index analysis
        include_secondary_mode: bool
            whether or not (default: False) to include secondary explosive mode analysis

        Returns
        -------
        output_library : cantera.SolutionArray instance
            The library with the following fields added (if requested)

            - 'cema-lexp1': the primary explosive eigenvalue (always computed)
            - 'cema-lexp2': the secondary explosive eigenvalue (default: off)
            - 'cema-ei1 [name]': the primary explosion indices for species (name) or temperature (T) (default: off)
            - 'cema-ei2 [name]': the secondary explosion indices for species (name) or temperature (T) (default: off)
            - 'cema-pi1 [#]': the primary participation indices of each reaction (numbered) (default: off)
            - 'cema-pi2 [#]': the secondary participation indices of each reaction (numbered) (default: off)
    """

    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" to explosive_mode_analysis() must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" to explosive_mode_analysis() must be a Library')

    gas = mechanism.gas
    griffon = mechanism.griffon
    if cantera_version_check('pre', 2, 6, None):
        V_stoich = gas.product_stoich_coeffs() - gas.reactant_stoich_coeffs()
    else:
        V_stoich = gas.product_stoich_coeffs3 - gas.reactant_stoich_coeffs3
    ns = gas.n_species
    ne = ns if configuration == 'isobaric' else ns + 1
    nr = mechanism.n_reactions
    Tidx = 0 if configuration == 'isobaric' else 1

    cema_Vmod = np.zeros((ne, nr))
    cema_Vmod[Tidx + 1:, :] = np.copy(V_stoich[:-1, :])
    cema_Vmod[Tidx, :] = (gas.standard_enthalpies_RT * ct.gas_constant * 298.).T.dot(V_stoich)

    library_shape = output_library['temperature'].shape
    nstates = output_library['temperature'].size

    lexp1 = np.zeros(nstates)
    if compute_explosion_indices:
        ei1_list = np.zeros((ns, nstates))
    if compute_participation_indices:
        pi1_list = np.zeros((nr, nstates))
    if include_secondary_mode:
        lexp2 = np.zeros(nstates)
        if compute_explosion_indices:
            ei2_list = np.zeros((ns, nstates))
        if compute_participation_indices:
            pi2_list = np.zeros((nr, nstates))

    Tflat = output_library['temperature'].ravel()
    Yflat = []
    for name in mechanism.species_names:
        Yflat.append(output_library['mass fraction ' + name].ravel())

    if configuration == 'isochoric':
        rhoflat = output_library['density'].ravel()

    if configuration == 'isobaric':
        pflat = output_library['pressure'].ravel()

    for i in range(nstates):
        # get the state
        state = np.zeros(ne)
        if configuration == 'isobaric':
            p = pflat[i]
            state[0] = Tflat[i]
            for ispec in range(ns - 1):
                state[1 + ispec] = Yflat[ispec][i]
        elif configuration == 'isochoric':
            state[0] = rhoflat[i]
            state[1] = Tflat[i]
            for ispec in range(ns - 1):
                state[2 + ispec] = Yflat[ispec][i]

        # get an appropriate Jacobian
        jac = np.zeros(ne * ne)
        null = np.zeros(ne)
        if configuration == 'isobaric':
            if heat_transfer == 'adiabatic' or heat_transfer == 'diathermal':
                griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                             0, 0, 0, False, 0, 0, null, jac)
                jac = jac.reshape((ne, ne), order='F')
            elif heat_transfer == 'isothermal':
                griffon.reactor_jac_isobaric(state, p, 0, np.ndarray(1), 0, 0, 0, 0,
                                             0, 0, 1, False, 0, 0, null, jac)
                jac = jac.reshape((ne, ne), order='F')

        elif configuration == 'isochoric':
            if heat_transfer == 'adiabatic' or heat_transfer == 'diathermal':
                griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                              0, 0, 0, 0, 0, 0, 0, False, 0, null, jac)
                jac = jac.reshape((ne, ne), order='F')
            elif heat_transfer == 'isothermal':
                griffon.reactor_jac_isochoric(state, 0, 0, np.ndarray(1),
                                              0, 0, 0, 0, 0, 0, 1, False, 0, null, jac)
                jac = jac.reshape((ne, ne), order='F')

        # do cema with the Jacobian
        if compute_explosion_indices or compute_participation_indices:
            w, vl, vr = eig(jac, left=True, right=True)
        else:
            w = eigvals(jac)

        realparts = np.real(w)
        threshold = 1.e-4
        realparts[(realparts > -threshold) & (realparts < threshold)] = -np.inf
        exp_idx1 = np.argmax(realparts)
        real_parts_without_1 = np.delete(realparts, exp_idx1)
        exp_idx2 = np.argmax(real_parts_without_1)
        lexp1[i] = realparts[exp_idx1]
        if include_secondary_mode:
            lexp2[i] = realparts[exp_idx2]

        if compute_explosion_indices or compute_participation_indices:
            exp_rvec1 = vr[:, exp_idx1]
            exp_lvec1 = vl[:, exp_idx1]
            exp_rvec2 = vr[:, exp_idx2]
            exp_lvec2 = vl[:, exp_idx2]
            ep_list1 = np.abs(np.real(exp_lvec1) * np.real(exp_rvec1))
            ei1_list[:, i] = ep_list1[Tidx:] / np.sum(np.abs(ep_list1))
            if include_secondary_mode:
                ep_list2 = np.abs(np.real(exp_lvec2) * np.real(exp_rvec2))
                ei2_list[:, i] = ep_list2[Tidx:] / (
                    1. if np.sum(np.abs(ep_list2)) < 1.e-16 else np.sum(np.abs(ep_list2)))

        if compute_participation_indices:
            if configuration == 'isobaric':
                ynm1 = state[1:]
                gas.TPY = state[0], pflat[i], np.hstack((ynm1, 1. - sum(ynm1)))
            elif configuration == 'isochoric':
                ynm1 = state[2:]
                gas.TDY = state[1], state[0], np.hstack((ynm1, 1. - sum(ynm1)))
            qnet = gas.net_rates_of_progress
            pi1 = np.abs(exp_lvec1.dot(cema_Vmod) * qnet)
            pi1_list[:, i] = pi1 / np.sum(np.abs(pi1))
            if include_secondary_mode:
                pi2 = np.abs(exp_lvec2.dot(cema_Vmod) * qnet)
                pi2_list[:, i] = pi2 / np.sum(np.abs(pi2))

    # insert quantities into the library
    output_library['cema-lexp1'] = lexp1.reshape(library_shape)
    if compute_explosion_indices:
        output_library['cema-ei1 T'] = ei1_list[0].reshape(library_shape)
        for ispec in range(ns - 1):
            output_library['cema-ei1 ' + mechanism.species_names[ispec]] = ei1_list[1 + ispec].reshape(library_shape)
    if compute_participation_indices:
        for ireac in range(nr):
            output_library['cema-pi1 ' + str(ireac)] = pi1_list[ireac].reshape(library_shape)
    if include_secondary_mode:
        output_library['cema-lexp2'] = lexp2.reshape(library_shape)
        if compute_explosion_indices:
            output_library['cema-ei2 T'] = ei2_list[0].reshape(library_shape)
            for ispec in range(ns - 1):
                output_library['cema-ei2 ' + mechanism.species_names[ispec]] = ei2_list[1 + ispec].reshape(library_shape)
        if compute_participation_indices:
            for ireac in range(nr):
                output_library['cema-pi2 ' + str(ireac)] = pi2_list[ireac].reshape(library_shape)
    return output_library
