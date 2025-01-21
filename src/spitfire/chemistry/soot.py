from spitfire import get_ct_solution_array
from spitfire.chemistry.analysis import compute_density
import numpy as np
from cantera import avogadro as Na
from cantera import gas_constant as Ru
from cantera import stefan_boltzmann as sigma
from cantera import boltzmann as kb
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


def _get_default_species_aliases():
    default_species_aliases = {
    'methane': 'CH4',
    'carbon-dioxide': 'CO2',
    'carbon-monoxide': 'CO',
    'water': 'H2O',
    'hydroxyl': 'OH',
    'acetylene': 'C2H2',
    'oxygen': 'O2',
    'monatomic-oxygen': 'O',
    'benzene': 'A1XC6H6',
    'naphthalene': 'A2XC10H8',
    'acenaphthalene': 'A2R5',
    'biphenyl': 'P2',
    'phenanthrene': 'A3',
    'acephenanthrylene': 'A3R5',
    'pyrene': 'A4',
    'flouranthene': 'FLTN',
    'cyclopentapyrene': 'A4R5'
    }
    return default_species_aliases


def tabulate_soot_source_coefficients(mechanism, output_library, modelname, percent_soot_leakage=0., verbose=True):
    """Adds soot source term coefficients to a library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        modelname: str
            the soot kinetics model. Supported models include: 'Aksit_Moss', 'Aksit_Moss_Benzene', and 'PAH_alpha' (an alpha-stage model)
        percent_soot_leakage: float
            the percent of soot allowed to leak into fuel-lean regions of the flame.
            This is similar to the ideas presented in (H.M. Colmán, A. Attili, M.E. Mueller,
            Large Eddy Simulation of turbulent nonpremixed sooting flames:
            presumed subfilter PDF model for finite-rate oxidation of soot, Combust. Flame, 258 (2023) 112602.).

        Returns
        -------
        output_library: a spitfire.chemistry.library Library instance
            The library with the following fields added for species relevant to the chosen model:

            - 'coag': coagulation
            - 'surf_': surface growth or oxidation for the species following the underscore
            - 'nucl_': nucleation for the species following the underscore
    """
    soot_species_aliases = _get_default_species_aliases()
    rhos = 1800.
    if 'density' not in output_library.props:
        compute_density(mechanism, output_library)
    rho = output_library['density']
    T = output_library['temperature']
    def mod_arrhenius_linear(A, Tpow, Tact, spec_name):
        return A * np.power(T, Tpow) * np.exp(-Tact / T) * output_library[f'mass fraction {spec_name}'] * rho / mechanism.molecular_weight(spec_name)

    def check_req_specs(req_spec_list):
        for nm in req_spec_list:
            if f'mass fraction {nm}' not in output_library.props:
                raise ValueError(f'Soot model {modelname} requires library with mass fraction {nm}.')

    if modelname == 'Aksit_Moss':
        check_req_specs([soot_species_aliases['acetylene'], soot_species_aliases['oxygen'], soot_species_aliases['hydroxyl']])
        multiplier = 1. - np.exp(16.e3 * (1.0 / 2050.0 - 1.0 / T))
        multiplier = np.maximum(multiplier, 0.)

        vol_fac = (6. / np.pi / rhos / Na) ** (1. / 3.)
        area_fac = vol_fac * vol_fac * np.pi * Na
        output_library['coag'] = np.sqrt(vol_fac * 24. * T * Ru / rhos * Na) * rho ** 2
        output_library['nucl_C2H2'] = multiplier * mod_arrhenius_linear(54.0, 0, 21.1e3, soot_species_aliases['acetylene'])
        output_library['surf_C2H2'] = rho * area_fac * multiplier * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['acetylene'])
        output_library['surf_O2'] = rho * area_fac * mod_arrhenius_linear(500., 0.5, 20.e3, soot_species_aliases['oxygen'])
        output_library['surf_OH'] = rho * area_fac * mod_arrhenius_linear(4.2325, 0.5, 0.0, soot_species_aliases['hydroxyl'])

    elif modelname == 'Aksit_Moss_Benzene':
        check_req_specs([soot_species_aliases['benzene'], soot_species_aliases['acetylene'], soot_species_aliases['oxygen'], soot_species_aliases['hydroxyl']])
        multiplier = 1. - np.exp(16.e3 * (1.0 / 2050.0 - 1.0 / T))
        multiplier = np.maximum(multiplier, 0.)
        nucleation_factor = 2.0
        A1_factor = 20.0

        vol_fac = (6. / np.pi / rhos / Na) ** (1. / 3.)
        area_fac = vol_fac * vol_fac * np.pi * Na
        output_library['coag'] = np.sqrt(vol_fac * 24. * T * Ru / rhos * Na) * rho ** 2
        output_library['nucl_C2H2'] = nucleation_factor * multiplier * mod_arrhenius_linear(54.0, 0, 21.1e3, soot_species_aliases['acetylene'])
        output_library['surf_C2H2'] = rho * area_fac * multiplier * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['acetylene'])
        output_library['surf_A1'] = A1_factor * multiplier * rho * area_fac * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['benzene'])
        output_library['surf_O2'] = rho * area_fac * mod_arrhenius_linear(500., 0.5, 20.e3, soot_species_aliases['oxygen'])
        output_library['surf_OH'] = rho * area_fac * mod_arrhenius_linear(4.2325, 0.5, 0.0, soot_species_aliases['hydroxyl'])
    elif modelname == 'PAH_alpha':
        if verbose:
            print('WARNING! The "PAH_alpha" soot model is an alpha-stage model not ready for use in calculations of consequence. \
While it has shown reasonable behavior on small-scale heptane and toluene pool fires in recent studies, \
it is associated with very little credibility evidence and had not undergone sufficient peer review for use in important calculations.')
        ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)
        MC = 12
        mmw = ctsol.mean_molecular_weight.reshape(lib_shape)
        conc = rho / mmw
        molar_conc = dict()
        mole_frac = dict()
        for s in mechanism.species_names:
            molar_conc[s] = rho * output_library[f'mass fraction {s}'] / mechanism.molecular_weight(s)
            mole_frac[s] = molar_conc[s] / conc  
        pah_nms = {k: soot_species_aliases[k] for k in ['naphthalene', 'acenaphthalene', 'biphenyl', 'phenanthrene', 'acephenanthrylene', 'pyrene', 'flouranthene', 'cyclopentapyrene']}
        check_req_specs(['O','OH'] + list(pah_nms.values()))
        pah_data = {'naphthalene': {'sticking': 0.001},
                    'acenaphthalene': {'sticking': 0.003},
                    'biphenyl': {'sticking': 0.0085},
                    'phenanthrene': {'sticking': 0.015},
                    'acephenanthrylene': {'sticking': 0.025},
                    'pyrene': {'sticking': 0.025},
                    'flouranthene': {'sticking': 0.025},
                    'cyclopentapyrene': {'sticking': 0.039}}
        for s in pah_nms:
            pah_data[s]['mw'] = mechanism.molecular_weight(pah_nms[s])  # kg/kmol
            Mi = pah_data[s]['mw']  # kg/kmol 
            pah_data[s]['nC'] = mechanism.gas.species(pah_nms[s]).composition['C']
            molar_conc[s] = molar_conc[pah_nms[s]]
        CD = np.mean([2. * pah_data[pah]['nC'] for pah in pah_data])
        MD = np.mean([2. * pah_data[pah]['mw'] for pah in pah_data])
        mD = MD / Na
        vD = mD / rhos
        dD = np.power(6.0 * mD / (np.pi * rhos), 1. / 3.)
        di = dD
        dj = dD
        mi = MD / MC
        mj = MD / MC
        mc = MC / Na
        miot = np.power(mi, 1. / 3.)
        mjot = np.power(mj, 1. / 3.)
        b1f = 2.2 * np.sqrt(6. * kb * T / rhos)
        b2f = np.power(3 * mc / (4. * np.pi * rhos), 1. / 6.)
        b3f = np.sqrt(1. / mi + 1. / mj)
        b4f = (miot + mjot) ** 2
        betaN = b1f * b2f * b3f * b4f
        Gm = 1. / Na
        Gv = 1. / (Na * rhos)
        Gd = np.power(6. / (np.pi * rhos * Na), 1./3.)
        GA = rho * Na * Gd * Gd
        GsA = np.sqrt(rho * Na) * Gd
        GV = rho / rhos

        beta = dict()
        dimer_rates = dict()
        omegaD = np.zeros_like(T)
        for pah in pah_nms:
            stick = pah_data[pah]['sticking']
            Mi = pah_data[pah]['mw']
            mi = Mi / Na
            dPAH = np.power(6.0 * mi / (np.pi * rhos), 1.0 / 3.0)
            beta[pah] = 2.2 * np.sqrt(4. * np.pi * kb * T / (MC / Na * pah_data[pah]['nC'])) * di * di
            beta[pah] = 2.2 * np.sqrt(4. * np.pi * Ru * T / (MC * pah_data[pah]['nC'])) * di * di

            ci  = molar_conc[pah_nms[pah]]
            nci = ci * Na
            
            collision_rate = beta[pah] * nci * nci
            stick_rate     = stick * collision_rate
            
            molar_rate     = stick_rate / Na
            mass_rate      = molar_rate * 2 * Mi
            
            dimer_rates[pah] = mass_rate
            omegaD += dimer_rates[pah]

        arrhenius = lambda A, n, E: A * np.power(T, n) * np.exp(-E / Ru / T)

        k1f = arrhenius(6.72e-2, 3.33, 6.09e6)
        k1b = arrhenius(6.44e-4, 3.79, 27.96e6)
        k2f = arrhenius(1.08e5, 1.80, 68.42e6)
        k2b = arrhenius(8.68e1, 2.36, 25.46e6)
        k3f = arrhenius(1.13e16, -0.06, 476.05e6)
        k3b = arrhenius(4.17e10, 0.15, 0.0)
        k4  = arrhenius(2.52e6, 1.1, 17.13e6)
        dr  = 0.0 
        fstar = 1.0 / (1.0 + (k1b * molar_conc['H2O'] + k2b * molar_conc['H2'] + k3b * molar_conc['H'] + k4 * molar_conc['C2H2']) 
                            / (k1f * molar_conc['OH'] + k2f * molar_conc['H'] + k3f + dr * k4))
        chi = 1.7e19
        zeta_haca = chi * fstar * k4 * molar_conc['C2H2'] * 2 * MC / Na

        kO2 = arrhenius(2.2e6, 0, 31.38e6)
        zeta_O2 = chi * kO2 * molar_conc['O2'] * 2 * MC / Na
        MOH = mechanism.molecular_weight('OH')
        OHeff = 0.04
        AOH = OHeff * np.sqrt(Ru / (2. * np.pi * MOH)) * MC
        MO = mechanism.molecular_weight('O')
        Oeff = 0.2
        AO = Oeff * np.sqrt(Ru / (2. * np.pi * MO)) * MC
        MO = mechanism.molecular_weight('O')
        Oeff = 0.2
        AO = Oeff * np.sqrt(Ru / (2. * np.pi * MO)) * MC
        coag = np.sqrt(24. * Ru * T * Na / rhos * Gd) * rho * rho

        output_library['coag'] = coag
        output_library['ox_O'] = AO * GA * np.sqrt(T) * molar_conc['O']
        output_library['ox_OH'] = AOH * GA * np.sqrt(T) * molar_conc['OH']
        output_library['ox_O2'] = zeta_O2 * GA
        output_library['sg'] = zeta_haca * GA
        output_library['beta_n'] = betaN
        output_library['beta_c_pre'] = 2.2 * np.sqrt(6. * kb * T / rhos) * np.power(3 * mc / (4. * np.pi * rhos), 1. / 6.)
        output_library['dimer_src'] = omegaD
    else:
        raise ValueError("Unsupported soot kinetics model.")

    if percent_soot_leakage>0.:
        if 'dissipation_rate_stoich' not in output_library.dim_names:
            raise ValueError("Soot leakage correction requires the library to have a 'dissipation_rate_stoich' dimension.")

        output_library['soot_correction'] = output_library.get_empty_dataset()
        growth_terms = ['surf_C2H2']
        if 'surf_A1' in output_library.props:
            growth_terms.append('surf_A1')
        ox_terms = ['surf_O2','surf_OH']
        coeff_sum = np.zeros(output_library.shape)
        kox = np.zeros(output_library.shape)
        for prop in growth_terms:
            coeff_sum += output_library[prop]
        for prop in ox_terms:
            coeff_sum -= output_library[prop]
            kox += output_library[prop]

        zvals = output_library.mixture_fraction_values
        ndims = len(output_library.dims)
        for ix in range(output_library.dissipation_rate_stoich_npts):
            def correction_1d(lib_slice):
                coeff_line = coeff_sum[lib_slice]
                kox_line = kox[lib_slice]
                func = lambda x: interp1d(zvals, coeff_line, kind=2)(x)
                iguess = np.argmax(func(zvals)>=0)-1
                zguess = zvals[iguess]
                zsoot = fsolve(func, zguess)[0]
                chi = output_library.dissipation_rate_stoich_values[ix]
                dz2 = 0.5 * chi / interp1d(zvals, kox_line, kind=3)(zsoot)
                return 0.5*(1.+percent_soot_leakage) + 0.5*(1.-percent_soot_leakage) * np.tanh((zvals - (zsoot - dz2)) / dz2)
            if ndims==2:
                lib_slice = tuple([slice(None), ix])
                output_library['soot_correction'][lib_slice] = correction_1d(lib_slice)
            elif ndims==3 and 'enthalpy_defect_stoich'==output_library.dim_names[-1]:
                for ig in range(output_library.enthalpy_defect_stoich_npts):
                    lib_slice = tuple([slice(None), ix, ig])
                    output_library['soot_correction'][lib_slice] = correction_1d(lib_slice)
            else:
                raise ValueError("Ensure your library is laminar.")

        for sgox in growth_terms + ox_terms:
            output_library[sgox] *= output_library['soot_correction']

    return output_library


def _linear_soot_radiation_model(temperature, rho, rhos, intercept, slope, min_value):
    return np.maximum(intercept + slope * temperature, min_value) * rho / rhos


def tabulate_soot_radiation_model(mechanism, output_library, modelname='linear_absorption_coefficient', input_modelargs=None):
    """Adds soot radiation properties to a library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        modelname: str
            the soot radiation model. Supported models include: 'linear_absorption_coefficient' (default)
        input_modelargs: dict
            a dictionary of arguments for the specified radiation model. For example, the 'linear_absorption_coefficient' model can take
            an 'intercept', 'slope', and 'min_value'.

        Returns
        -------
        output_library: a spitfire.chemistry.library Library instance
            The library with the following fields added

            - 'soot_abs_coeff': absorption coefficient
            - 'soot_rad_source': radiation source
    """
    if modelname=='linear_absorption_coefficient':
        if 'density' not in output_library.props:
            compute_density(mechanism, output_library)
        modelargs = dict({'intercept': 0.,
                          'slope': 0.,
                          'min_value': 0.})
        if input_modelargs is not None:
            modelargs.update(input_modelargs)
        output_library['soot_abs_coeff'] = _linear_soot_radiation_model(output_library['temperature'], output_library['density'], 1800., **modelargs)
        output_library['soot_rad_source'] = output_library['soot_abs_coeff'] * sigma / np.pi * np.power(output_library['temperature'], 4)
    else:
        raise ValueError("Unsupported soot radiation model.")
    return output_library


def _detect_soot_model(library, verbose):
    properties = library.props
    soot_kinetics = None
    soot_props = ['coag', 'nucl_C2H2', 'surf_C2H2', 'surf_O2', 'surf_OH']
    if all([sp in properties for sp in soot_props]):
        if all([sp in properties for sp in ['surf_A1']]):
            soot_kinetics = 'aksit-moss-benzene'
            soot_props.append('surf_A1')
        else:
            soot_kinetics = 'aksit-moss'
    else:
        soot_props = ['coag', 'dimer_src', 'sg', 'ox_O', 'ox_O2', 'ox_OH']
        if all([sp in properties for sp in soot_props]):
            soot_kinetics = 'pah-alpha'

    if soot_kinetics is not None:
        if verbose:
            print(f'Detected soot model "{soot_kinetics}" in the table.')
    else:
        raise ValueError(f'Could not detect soot model in the library.')
    return soot_kinetics, soot_props


def _get_default_mult_dict():
    return dict({'nucl': 1.0,
                  'coag': 1.0,
                  'cond': 1.0,
                  'c2h2': 1.0,
                  'o2': 1.0,
                  'oh': 1.0,
                  'o': 1.0,
                  'a1': 1.0})


def _soot_src_aksit_moss(ns, ys, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, input_mult_dict=None):
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(ns, 1./3.) * np.power(ys, 2./3.)
    nucl_mole_rate = mult_dict['nucl'] * nucl_C2H2
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(ns, 11./6.) * np.power(ys, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * 144.0 * nucl_C2H2
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['c2h2'] * surf_C2H2 * area_fac
    surf_o2_rate = -mult_dict['o2'] * surf_O2 * area_fac
    surf_oh_rate = -mult_dict['oh'] * surf_OH * area_fac
    surf_o_rate = np.zeros_like(surf_o2_rate)
    surf_a1_rate = np.zeros_like(surf_c2h2_rate)

    src_dict = dict({
        'S_Ns': nucl_mole_rate + coag_mole_rate,
        'S_Ys': nucl_mass_rate + cond_rate + surf_c2h2_rate + surf_a1_rate + surf_o2_rate + surf_oh_rate + surf_o_rate,
        'S_Ns_nucl': nucl_mole_rate,
        'S_Ns_coag': coag_mole_rate,
        'S_Ys_nucl': nucl_mass_rate,
        'S_Ys_cond': cond_rate,
        'S_Ys_c2h2': surf_c2h2_rate,
        'S_Ys_o2': surf_o2_rate,
        'S_Ys_oh': surf_oh_rate,
        'S_Ys_o': surf_o_rate,
        'S_Ys_a1': surf_a1_rate
    })
    return src_dict


def _soot_src_aksit_moss_benzene(ns, ys, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, surf_A1, input_mult_dict=None):
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(ns, 1./3.) * np.power(ys, 2./3.)
    nucl_mole_rate = mult_dict['nucl'] * nucl_C2H2
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(ns, 11./6.) * np.power(ys, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * 144.0 * nucl_C2H2
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['c2h2'] * surf_C2H2 * area_fac
    surf_o2_rate = -mult_dict['o2'] * surf_O2 * area_fac
    surf_oh_rate = -mult_dict['oh'] * surf_OH * area_fac
    surf_o_rate = np.zeros_like(surf_o2_rate)
    surf_a1_rate = mult_dict['a1'] * surf_A1 * area_fac

    src_dict = dict({
        'S_Ns': nucl_mole_rate + coag_mole_rate,
        'S_Ys': nucl_mass_rate + cond_rate + surf_c2h2_rate + surf_a1_rate + surf_o2_rate + surf_oh_rate + surf_o_rate,
        'S_Ns_nucl': nucl_mole_rate,
        'S_Ns_coag': coag_mole_rate,
        'S_Ys_nucl': nucl_mass_rate,
        'S_Ys_cond': cond_rate,
        'S_Ys_c2h2': surf_c2h2_rate,
        'S_Ys_o2': surf_o2_rate,
        'S_Ys_oh': surf_oh_rate,
        'S_Ys_o': surf_o_rate,
        'S_Ys_a1': surf_a1_rate
    })
    return src_dict

def _soot_src_pah_alpha(ns, ys, dimer_src, coag, sg, ox_O2, ox_OH, ox_O, input_mult_dict=None):
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(ns, 1./3.) * np.power(ys, 2./3.)
    MD = 361.4
    dimer_fac = 1.e-6
    nucl_mole_rate = mult_dict['nucl'] * dimer_fac * dimer_src * 0.5 / MD
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(ns, 11./6.) * np.power(ys, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * dimer_fac * dimer_src
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['c2h2'] * sg * area_fac
    surf_o2_rate = -mult_dict['o2'] * ox_O2 * area_fac
    surf_oh_rate = -mult_dict['oh'] * ox_OH * area_fac
    surf_o_rate = -mult_dict['o'] * ox_O * area_fac
    surf_a1_rate = np.zeros_like(surf_c2h2_rate)

    src_dict = dict({
        'S_Ns': nucl_mole_rate + coag_mole_rate,
        'S_Ys': nucl_mass_rate + cond_rate + surf_c2h2_rate + surf_a1_rate + surf_o2_rate + surf_oh_rate + surf_o_rate,
        'S_Ns_nucl': nucl_mole_rate,
        'S_Ns_coag': coag_mole_rate,
        'S_Ys_nucl': nucl_mass_rate,
        'S_Ys_cond': cond_rate,
        'S_Ys_c2h2': surf_c2h2_rate,
        'S_Ys_o2': surf_o2_rate,
        'S_Ys_oh': surf_oh_rate,
        'S_Ys_o': surf_o_rate,
        'S_Ys_a1': surf_a1_rate
    })
    return src_dict


def compute_soot_source_terms(library, soot_number_density, soot_mass_fraction, clip=True, input_mult_dict=None, verbose=True):
    soot_kinetics, soot_props = _detect_soot_model(library, verbose)
    ns_clip = np.minimum(np.maximum(soot_number_density, 1.e-16), 1.e6) if clip else soot_number_density
    ys_clip = np.minimum(np.maximum(soot_mass_fraction, 1.e-16), 1.) if clip else soot_mass_fraction

    if ns_clip<0. or ys_clip<0.:
        raise ValueError("Cannot have negative soot_number_density or soot_mass_fraction.")

    if soot_kinetics=='aksit-moss':
        src_dict = _soot_src_aksit_moss(ns_clip, ys_clip, library['nucl_C2H2'], library['coag'], library['surf_C2H2'], library['surf_O2'], library['surf_OH'], input_mult_dict)
    elif soot_kinetics=='aksit-moss-benzene':
        src_dict = _soot_src_aksit_moss_benzene(ns_clip, ys_clip, library['nucl_C2H2'], library['coag'], library['surf_C2H2'], library['surf_O2'], library['surf_OH'], library['surf_A1'], input_mult_dict)
    elif soot_kinetics=='pah-alpha':
        src_dict = _soot_src_pah_alpha(ns_clip, ys_clip, library['dimer_src'], library['coag'], library['sg'], library['ox_O2'], library['ox_OH'], library['ox_O'], input_mult_dict)
    return src_dict
