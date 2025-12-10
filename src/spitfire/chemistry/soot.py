from spitfire import get_ct_solution_array, compute_dissipation_rate, Library
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


def tabulate_soot_source_coefficients(mechanism, input_library, modelname, soot_leakage_fraction=1., verbose=True, soot_species_aliases=None, soot_density=1800., AMB_nucleation_factor=2., AMB_A1_factor=20.):
    """Adds soot source term coefficients to a library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        input_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        modelname: str
            the soot kinetics model. Supported models include: 'Aksit_Moss', 'Aksit_Moss_Benzene', and 'PAH_alpha' (an alpha-stage model)
        soot_leakage_fraction: float
            the fraction of soot allowed to leak into fuel-lean regions of the flame.
        verbose: bool
            whether or not to print warnings/messages (default True)
        soot_species_aliases: dictionary
            dictionary of aliases for the soot species involved, matching the species name to its abbreviation/formula used in the mechanism.
            Spitfire has some common aliases it checks if none are provided.
        soot_density: float
            density of soot (default = 1800 kg/m^3)
        AMB_nucleation_factor: float
            multiplier on the 'Aksit_Moss_Benzene' nucleation (default 2)
        AMB_A1_factor: float
            multiplier on the 'Aksit_Moss_Benzene' A1 surface growth (default 20)

        Returns
        -------
        output_library: a spitfire.chemistry.library Library instance
            The library with the following fields added for species relevant to the chosen model:

            - 'coag': coagulation
            - 'surf\_': surface growth or oxidation for the species following the underscore
            - 'nucl\_': nucleation for the species following the underscore
    """
    output_library = Library.copy(input_library)
    soot_species_aliases = soot_species_aliases if soot_species_aliases is not None else _get_default_species_aliases()
    if 'density' not in output_library.props:
        compute_density(mechanism, output_library)
    rho = output_library['density']
    T = output_library['temperature']
    def mod_arrhenius_linear(A, Tpow, Tact, spec_name):
        return A * np.power(T, Tpow) * np.exp(-Tact / T) * output_library[f'mass fraction {spec_name}'] * rho / mechanism.molecular_weight(spec_name)

    def check_req_specs(req_spec_list, alias_dict):
        for sp in req_spec_list:
            if sp not in alias_dict:
                raise ValueError(f"Soot model {modelname} requires {sp}. This key must be present in the soot_species_aliases dictionary and point to the corresponding species formula.")
            nm = alias_dict[sp]
            if f'mass fraction {nm}' not in output_library.props:
                msg = f'Soot model {modelname} requires {sp}. Looked in library for mass fraction {nm}, but did not find it.'
                msg += f'\nIf this mass fraction species is incorrect, provide the correct species through the soot_species_aliases argument for key {sp}.'
                raise ValueError(msg)
    if modelname == 'Aksit_Moss':
        check_req_specs(['acetylene', 'oxygen', 'hydroxyl'], soot_species_aliases)
        multiplier = 1. - np.exp(16.e3 * (1.0 / 2050.0 - 1.0 / T))
        multiplier = np.maximum(multiplier, 0.)

        vol_fac = (6. / np.pi / soot_density / Na) ** (1. / 3.)
        area_fac = vol_fac * vol_fac * np.pi * Na
        output_library['coag'] = np.sqrt(vol_fac * 24. * T * Ru / soot_density * Na) * rho ** 2
        output_library['nucl_C2H2'] = multiplier * mod_arrhenius_linear(54.0, 0, 21.1e3, soot_species_aliases['acetylene'])
        output_library['surf_C2H2'] = rho * area_fac * multiplier * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['acetylene'])
        output_library['surf_O2'] = rho * area_fac * mod_arrhenius_linear(500., 0.5, 20.e3, soot_species_aliases['oxygen'])
        output_library['surf_OH'] = rho * area_fac * mod_arrhenius_linear(4.2325, 0.5, 0.0, soot_species_aliases['hydroxyl'])

    elif modelname == 'Aksit_Moss_Benzene':
        check_req_specs(['benzene', 'acetylene', 'oxygen', 'hydroxyl'], soot_species_aliases)
        multiplier = 1. - np.exp(16.e3 * (1.0 / 2050.0 - 1.0 / T))
        multiplier = np.maximum(multiplier, 0.)

        vol_fac = (6. / np.pi / soot_density / Na) ** (1. / 3.)
        area_fac = vol_fac * vol_fac * np.pi * Na
        output_library['coag'] = np.sqrt(vol_fac * 24. * T * Ru / soot_density * Na) * rho ** 2
        output_library['nucl_C2H2'] = AMB_nucleation_factor * multiplier * mod_arrhenius_linear(54.0, 0, 21.1e3, soot_species_aliases['acetylene'])
        output_library['surf_C2H2'] = rho * area_fac * multiplier * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['acetylene'])
        output_library['surf_A1'] = AMB_A1_factor * multiplier * rho * area_fac * mod_arrhenius_linear(11.7e3, 0, 12.e3, soot_species_aliases['benzene'])
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
        pah_nms_list = ['naphthalene', 'acenaphthalene', 'biphenyl', 'phenanthrene', 'acephenanthrylene', 'pyrene', 'flouranthene', 'cyclopentapyrene']
        pah_nms = {k: soot_species_aliases[k] for k in pah_nms_list}
        if 'O' not in soot_species_aliases:
            soot_species_aliases['O'] = 'O'
        if 'OH' not in soot_species_aliases:
            soot_species_aliases['OH'] = 'OH'
        check_req_specs(['O', 'OH']+pah_nms_list, soot_species_aliases)
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
        vD = mD / soot_density
        dD = np.power(6.0 * mD / (np.pi * soot_density), 1. / 3.)
        di = dD
        dj = dD
        mi = MD / MC
        mj = MD / MC
        mc = MC / Na
        miot = np.power(mi, 1. / 3.)
        mjot = np.power(mj, 1. / 3.)
        b1f = 2.2 * np.sqrt(6. * kb * T / soot_density)
        b2f = np.power(3 * mc / (4. * np.pi * soot_density), 1. / 6.)
        b3f = np.sqrt(1. / mi + 1. / mj)
        b4f = (miot + mjot) ** 2
        betaN = b1f * b2f * b3f * b4f
        Gm = 1. / Na
        Gv = 1. / (Na * soot_density)
        Gd = np.power(6. / (np.pi * soot_density * Na), 1./3.)
        GA = rho * Na * Gd * Gd
        GsA = np.sqrt(rho * Na) * Gd
        GV = rho / soot_density

        beta = dict()
        dimer_rates = dict()
        omegaD = np.zeros_like(T)
        for pah in pah_nms:
            stick = pah_data[pah]['sticking']
            Mi = pah_data[pah]['mw']
            mi = Mi / Na
            dPAH = np.power(6.0 * mi / (np.pi * soot_density), 1.0 / 3.0)
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
        coag = np.sqrt(24. * Ru * T * Na / soot_density * Gd) * rho * rho

        output_library['coag'] = coag
        output_library['ox_O'] = AO * GA * np.sqrt(T) * molar_conc['O']
        output_library['ox_OH'] = AOH * GA * np.sqrt(T) * molar_conc['OH']
        output_library['ox_O2'] = zeta_O2 * GA
        output_library['sg'] = zeta_haca * GA
        output_library['beta_n'] = betaN
        output_library['beta_c_pre'] = 2.2 * np.sqrt(6. * kb * T / soot_density) * np.power(3 * mc / (4. * np.pi * soot_density), 1. / 6.)
        output_library['dimer_src'] = omegaD
    else:
        raise ValueError("Unsupported soot kinetics model.")

    if soot_leakage_fraction>=0. and soot_leakage_fraction<1.:
        if 'dissipation_rate_stoich' not in output_library.dim_names:
            raise ValueError("Soot leakage correction requires the library to have a 'dissipation_rate_stoich' dimension.")

        output_library['soot_subfilter'] = output_library.get_empty_dataset()
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
        fuel = output_library.extra_attributes['fuel_stream']
        oxidizer = output_library.extra_attributes['oxy_stream']
        zst = mechanism.stoich_mixture_fraction(fuel, oxidizer)
        diss_rate_form = output_library.extra_attributes['dissipation_rate_form']
        for ix in range(output_library.dissipation_rate_stoich_npts):
            def correction_1d(lib_slice):
                coeff_line = coeff_sum[lib_slice]
                kox_line = kox[lib_slice]
                func = lambda x, order: interp1d(zvals, coeff_line, kind=order)(x)
                iguess = np.argmax(func(zvals, 2)>=0)-1
                zguess = zvals[iguess]
                try:
                    zsoot = fsolve(func, zguess, args=(2))[0]
                except:
                    zsoot = fsolve(func, zguess, args=(1))[0]
                if zsoot<1.e-12 or zsoot>1.-1.e-12:
                    return None
                else:
                    chi_st = output_library.dissipation_rate_stoich_values[ix]
                    chi = compute_dissipation_rate(zsoot, form=diss_rate_form, stoich_dissipation_rate=chi_st, stoich_mixture_fraction=zst)
                    dz2 = 0.5 * chi / interp1d(zvals, kox_line, kind=3)(zsoot)
                    return 0.5*(1.+soot_leakage_fraction) + 0.5*(1.-soot_leakage_fraction) * np.tanh((zvals - (zsoot - dz2)) / dz2)
            if ndims==2:
                lib_slice = tuple([slice(None), ix])
                subfilter = correction_1d(lib_slice)
                if subfilter is None and verbose:
                    print(f"Couldn't find Zsoot for index ({ix}): chi_st={output_library.dissipation_rate_stoich_values[ix]:.2e}. Will use a subfilter of 1.")
                output_library['soot_subfilter'][lib_slice] = np.ones_like(zvals) if subfilter is None else subfilter
            elif ndims==3 and 'enthalpy_defect_stoich'==output_library.dim_names[-1]:
                for ig in range(output_library.enthalpy_defect_stoich_npts):
                    lib_slice = tuple([slice(None), ix, ig])
                    subfilter = correction_1d(lib_slice)
                    if subfilter is None and verbose:
                        print(f"Couldn't find Zsoot for index ({ix}, {ig}): (chi_st={output_library.dissipation_rate_stoich_values[ix]:.2e}, g_st={output_library.enthalpy_defect_stoich_values[ig]:.2e}). Will use a subfilter of 1.")
                    output_library['soot_subfilter'][lib_slice] = np.ones_like(zvals) if subfilter is None else subfilter
            else:
                raise ValueError("Ensure your library is laminar.")

        for sgox in growth_terms + ox_terms:
            output_library[sgox] *= output_library['soot_subfilter']

    return output_library


def _linear_soot_radiation_model(temperature, rho, soot_density, intercept, slope, min_value):
    return np.maximum(intercept + slope * temperature, min_value) * rho / soot_density


def tabulate_soot_radiation_model(mechanism, input_library, modelname='linear_absorption_coefficient', input_modelargs=None, soot_density=1800.):
    """Adds soot radiation properties to a library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        input_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        modelname: str
            the soot radiation model. Supported models include: 'linear_absorption_coefficient' (default)
        input_modelargs: dictionary
            a dictionary of arguments for the specified radiation model. For example, the 'linear_absorption_coefficient' model can take
            an 'intercept' (default: -3.75e5 1/m), 'slope' (default: 1735. 1/(m-K)), and 'min_value' (default: 0. 1/m).
        soot_density: float
            density of soot (default = 1800 kg/m^3)

        Returns
        -------
        output_library: a spitfire.chemistry.library Library instance
            The library with the following fields added

            - 'soot_abs_coeff': absorption coefficient
            - 'soot_rad_source': radiation source
    """
    output_library = Library.copy(input_library)
    if modelname=='linear_absorption_coefficient':
        if 'density' not in output_library.props:
            compute_density(mechanism, output_library)
        modelargs = dict({'intercept': -3.75e5,
                          'slope': 1735.,
                          'min_value': 0.})
        if input_modelargs is not None:
            modelargs.update(input_modelargs)
        output_library['soot_abs_coeff'] = _linear_soot_radiation_model(output_library['temperature'], output_library['density'], soot_density, **modelargs)
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
                  'surf_c2h2': 1.0,
                  'surf_o2': 1.0,
                  'surf_oh': 1.0,
                  'surf_o': 1.0,
                  'surf_a1': 1.0})


def soot_src_aksit_moss(soot_number_density, soot_mass_fraction, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, input_mult_dict=None):
    """
    Evaluates source terms using the Aksit_Moss model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\dot{S}_{N_s} = \\mathrm{nucl\_C2H2} - \\mathrm{coag} N_s^{11/6} Y_s^{1/6}`

    :math:`\\dot{S}_{Y_s} = 144\\mathrm{nucl\_C2H2} + (\\mathrm{surf\_C2H2} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{1/3} Y_s^{2/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        nucl_C2H2: float or array
            an acetylene-based nucleation coefficient (:math:`\\mathrm{kmol}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        surf_C2H2: float or array
            an acetylene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        src_dict: dictionary
            A dictionary of the evaluated source terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys') along with various contributions to
            those source terms including nucleation ('S_Ns_nucl' and 'S_Ys_nucl'), coagulation ('S_Ns_coag'), condensation ('S_Ys_cond'),
            surface growth ('S_Ys_c2h2' and 'S_Ys_a1'), and oxidation ('S_Ys_o2', 'S_Ys_oh', and 'S_Ys_o'). Terms that aren't relevant to the library's soot model 
            will be populated with zeros.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, 2./3.)
    nucl_mole_rate = mult_dict['nucl'] * nucl_C2H2
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * 144.0 * nucl_C2H2
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['surf_c2h2'] * surf_C2H2 * area_fac
    surf_o2_rate = -mult_dict['surf_o2'] * surf_O2 * area_fac
    surf_oh_rate = -mult_dict['surf_oh'] * surf_OH * area_fac
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


def soot_src_aksit_moss_benzene(soot_number_density, soot_mass_fraction, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, surf_A1, input_mult_dict=None):
    """
    Evaluates source terms using the Aksit_Moss_Benzene model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\dot{S}_{N_s} = \\mathrm{nucl\_C2H2} - \\mathrm{coag} N_s^{11/6} Y_s^{1/6}`

    :math:`\\dot{S}_{Y_s} = 144\\mathrm{nucl\_C2H2} + (\\mathrm{surf\_C2H2} + \\mathrm{surf\_A1} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{1/3} Y_s^{2/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        nucl_C2H2: float or array
            an acetylene-based nucleation coefficient (:math:`\\mathrm{kmol}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        surf_C2H2: float or array
            an acetylene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_A1: float or array
            a benzene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        src_dict: dictionary
            A dictionary of the evaluated source terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys') along with various contributions to
            those source terms including nucleation ('S_Ns_nucl' and 'S_Ys_nucl'), coagulation ('S_Ns_coag'), condensation ('S_Ys_cond'),
            surface growth ('S_Ys_c2h2' and 'S_Ys_a1'), and oxidation ('S_Ys_o2', 'S_Ys_oh', and 'S_Ys_o'). Terms that aren't relevant to the library's soot model 
            will be populated with zeros.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, 2./3.)
    nucl_mole_rate = mult_dict['nucl'] * nucl_C2H2
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * 144.0 * nucl_C2H2
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['surf_c2h2'] * surf_C2H2 * area_fac
    surf_o2_rate = -mult_dict['surf_o2'] * surf_O2 * area_fac
    surf_oh_rate = -mult_dict['surf_oh'] * surf_OH * area_fac
    surf_o_rate = np.zeros_like(surf_o2_rate)
    surf_a1_rate = mult_dict['surf_a1'] * surf_A1 * area_fac

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


def soot_src_pah_alpha(soot_number_density, soot_mass_fraction, dimer_src, coag, sg, ox_O2, ox_OH, ox_O, dimer_fac=1.e-6, input_mult_dict=None):
    """
    Evaluates source terms using the PAH_alpha model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\dot{S}_{N_s} = \\mathrm{dimer\_src}(\\mathrm{dimer\_fac})\\frac{0.5}{361.4} - \\mathrm{coag} N_s^{11/6} Y_s^{1/6}`

    :math:`\\dot{S}_{Y_s} = \\mathrm{dimer\_src}(\\mathrm{dimer\_fac}) + (\\mathrm{sg} - \\mathrm{ox\_O2} - \\mathrm{ox\_OH} - \\mathrm{ox\_O}) N_s^{1/3} Y_s^{2/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        dimer_src: float or array
            a PAH-based nucleation coefficient (:math:`\\mathrm{kg}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        sg: float or array
            a surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_O: float or array
            an oxygen radical-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        dimer_fac: float or array
            factor (default 1e-6) multiplying the nucleation coefficients; used to correct the model's tendency for nucleation to dominate and over-predict soot
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        src_dict: dictionary
            A dictionary of the evaluated source terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys') along with various contributions to
            those source terms including nucleation ('S_Ns_nucl' and 'S_Ys_nucl'), coagulation ('S_Ns_coag'), condensation ('S_Ys_cond'),
            surface growth ('S_Ys_c2h2' and 'S_Ys_a1'), and oxidation ('S_Ys_o2', 'S_Ys_oh', and 'S_Ys_o'). Terms that aren't relevant to the library's soot model 
            will be populated with zeros.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)
    area_fac = np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, 2./3.)
    MD = 361.4
    nucl_mole_rate = mult_dict['nucl'] * dimer_fac * dimer_src * 0.5 / MD
    coag_mole_rate = -mult_dict['coag'] * coag * np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, 1./6.)
    nucl_mass_rate = mult_dict['nucl'] * dimer_fac * dimer_src
    cond_rate = np.zeros_like(nucl_mass_rate)
    surf_c2h2_rate = mult_dict['surf_c2h2'] * sg * area_fac
    surf_o2_rate = -mult_dict['surf_o2'] * ox_O2 * area_fac
    surf_oh_rate = -mult_dict['surf_oh'] * ox_OH * area_fac
    surf_o_rate = -mult_dict['surf_o'] * ox_O * area_fac
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


def get_soot_source_term_evaluator(library, verbose=True):
    """
    Detect the soot model used in the library and return the corresponding source term function

        Parameters
        ----------
        library: a spitfire.chemistry.library Library instance
            a library with soot source term coefficients, computed from running tabulate_soot_source_coefficients
        verbose: boolean
            whether or not to print the detected soot model on a library; default True

        Returns
        -------
        A function for evaluating soot source terms
    """
    soot_kinetics, _ = _detect_soot_model(library, verbose)

    if soot_kinetics=='aksit-moss':
        return soot_src_aksit_moss
    elif soot_kinetics=='aksit-moss-benzene':
        return soot_src_aksit_moss_benzene
    elif soot_kinetics=='pah-alpha':
        return soot_src_pah_alpha


def jac_soot_src_aksit_moss(soot_number_density, soot_mass_fraction, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, input_mult_dict=None):
    """
    Evaluates components of the soot source Jacobian using the Aksit_Moss model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\partial{S}_{N_s}/\\partial{N_s} = - (11/6) \\mathrm{coag} N_s^{5/6} Y_s^{1/6}`

    :math:`\\partial{S}_{N_s}/\\partial{Y_s} = - (1/6) \\mathrm{coag} N_s^{11/6} Y_s^{-5/6}`

    :math:`\\partial{S}_{Y_s}/\\partial{N_s} = (1/3) (\\mathrm{surf\_C2H2} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{-2/3} Y_s^{2/3}`

    :math:`\\partial{S}_{Y_s}/\\partial{Y_s} = (2/3) (\\mathrm{surf\_C2H2} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{1/3} Y_s^{-1/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        nucl_C2H2: float or array
            an acetylene-based nucleation coefficient (:math:`\\mathrm{kmol}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        surf_C2H2: float or array
            an acetylene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        jac_dict: dictionary
            A dictionary of the evaluated source Jacobian terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys').
            Keys include 'dSNs_dNs', 'dSNs_dYs', 'dSYs_dNs', 'dSYs_dYs'.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)

    area_fac_Ns = (1./3.)*np.power(soot_number_density, -2./3.) * np.power(soot_mass_fraction, 2./3.)
    area_fac_Ys = (2./3.)*np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, -1./3.)

    coag_fac_Ns = (11./6.)*np.power(soot_number_density, 5./6.) * np.power(soot_mass_fraction, 1./6.)
    coag_fac_Ys = (1./6.)*np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, -5./6.)

    coag_mole_rate = -mult_dict['coag'] * coag
    surf_c2h2_rate = mult_dict['surf_c2h2'] * surf_C2H2
    surf_o2_rate = -mult_dict['surf_o2'] * surf_O2
    surf_oh_rate = -mult_dict['surf_oh'] * surf_OH

    dSNs_dNs = coag_mole_rate*coag_fac_Ns
    dSNs_dYs = coag_mole_rate*coag_fac_Ys
    dSYs_dNs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate)*area_fac_Ns
    dSYs_dYs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate)*area_fac_Ys

    jac_dict = {
        'dSNs_dNs': dSNs_dNs,
        'dSNs_dYs': dSNs_dYs,
        'dSYs_dNs': dSYs_dNs,
        'dSYs_dYs': dSYs_dYs,
    }
    return jac_dict


def jac_soot_src_aksit_moss_benzene(soot_number_density, soot_mass_fraction, nucl_C2H2, coag, surf_C2H2, surf_O2, surf_OH, surf_A1, input_mult_dict=None):
    """
    Evaluates components of the soot source Jacobian using the Aksit_Moss_Benzene model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\partial{S}_{N_s}/\\partial{N_s} = - (11/6) \\mathrm{coag} N_s^{5/6} Y_s^{1/6}`

    :math:`\\partial{S}_{N_s}/\\partial{Y_s} = - (1/6) \\mathrm{coag} N_s^{11/6} Y_s^{-5/6}`

    :math:`\\partial{S}_{Y_s}/\\partial{N_s} = (1/3) (\\mathrm{surf\_C2H2} + \\mathrm{surf\_A1} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{-2/3} Y_s^{2/3}`

    :math:`\\partial{S}_{Y_s}/\\partial{Y_s} = (2/3) (\\mathrm{surf\_C2H2} + \\mathrm{surf\_A1} - \\mathrm{surf\_O2} - \\mathrm{surf\_OH}) N_s^{1/3} Y_s^{-1/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        nucl_C2H2: float or array
            an acetylene-based nucleation coefficient (:math:`\\mathrm{kmol}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        surf_C2H2: float or array
            an acetylene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        surf_A1: float or array
            a benzene-based surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        jac_dict: dictionary
            A dictionary of the evaluated source Jacobian terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys').
            Keys include 'dSNs_dNs', 'dSNs_dYs', 'dSYs_dNs', 'dSYs_dYs'.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)

    area_fac_Ns = (1./3.)*np.power(soot_number_density, -2./3.) * np.power(soot_mass_fraction, 2./3.)
    area_fac_Ys = (2./3.)*np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, -1./3.)

    coag_fac_Ns = (11./6.)*np.power(soot_number_density, 5./6.) * np.power(soot_mass_fraction, 1./6.)
    coag_fac_Ys = (1./6.)*np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, -5./6.)

    coag_mole_rate = -mult_dict['coag'] * coag
    surf_c2h2_rate = mult_dict['surf_c2h2'] * surf_C2H2
    surf_o2_rate = -mult_dict['surf_o2'] * surf_O2
    surf_oh_rate = -mult_dict['surf_oh'] * surf_OH
    surf_a1_rate = mult_dict['surf_a1'] * surf_A1

    dSNs_dNs = coag_mole_rate*coag_fac_Ns
    dSNs_dYs = coag_mole_rate*coag_fac_Ys
    dSYs_dNs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate+surf_a1_rate)*area_fac_Ns
    dSYs_dYs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate+surf_a1_rate)*area_fac_Ys

    jac_dict = {
        'dSNs_dNs': dSNs_dNs,
        'dSNs_dYs': dSNs_dYs,
        'dSYs_dNs': dSYs_dNs,
        'dSYs_dYs': dSYs_dYs,
    }
    return jac_dict


def jac_soot_src_pah_alpha(soot_number_density, soot_mass_fraction, dimer_src, coag, sg, ox_O2, ox_OH, ox_O, dimer_fac=1.e-6, input_mult_dict=None):
    """
    Evaluates components of the soot source Jacobian using the PAH_alpha model for the soot_number_density (:math:`N_s`) and soot_mass_fraction (:math:`Y_s`) as shown below
    
    :math:`\\partial{S}_{N_s}/\\partial{N_s} = - (11/6) \\mathrm{coag} N_s^{5/6} Y_s^{1/6}`

    :math:`\\partial{S}_{N_s}/\\partial{Y_s} = - (1/6) \\mathrm{coag} N_s^{11/6} Y_s^{-5/6}`

    :math:`\\partial{S}_{Y_s}/\\partial{N_s} = (1/3) (\\mathrm{sg} - \\mathrm{ox\_O2} - \\mathrm{ox\_OH} - \\mathrm{ox\_O}) N_s^{-2/3} Y_s^{2/3}`

    :math:`\\partial{S}_{Y_s}/\\partial{Y_s} = (2/3) (\\mathrm{sg} - \\mathrm{ox\_O2} - \\mathrm{ox\_OH} - \\mathrm{ox\_O}) N_s^{1/3} Y_s^{-1/3}`

        Parameters
        ----------
        soot_number_density: float or array
            a soot number density (kmol soot / kg air)
        soot_mass_fraction: float or array
            a soot mass fraction
        dimer_src: float or array
            a PAH-based nucleation coefficient (:math:`\\mathrm{kg}/\\mathrm{m}^3/\\mathrm{s}`)
        coag: float or array
            a coagulation coefficient (:math:`\\mathrm{kg}^{11/6}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{5/6}`)
        sg: float or array
            a surface growth coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_O2: float or array
            an oxygen-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_OH: float or array
            a hydroxyl-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        ox_O: float or array
            an oxygen radical-based oxidation coefficient (:math:`\\mathrm{kg}^{4/3}/\\mathrm{m}^3/\\mathrm{s}/\\mathrm{kmol}^{1/3}`)
        dimer_fac: float or array
            factor (default 1e-6) multiplying the nucleation coefficients; used to correct the model's tendency for nucleation to dominate and over-predict soot
        input_mult_dict: dictionary
            a dictionary of multipliers for any of the source term coefficients present in a library. surface growth and oxidation multipliers
            are specified by the species name. Supported keys, showing the default unity multipliers, include
            {'nucl': 1.0, 'coag': 1.0, 'cond': 1.0, 'surf_c2h2': 1.0, 'surf_o2': 1.0, 'surf_oh': 1.0, 'surf_o': 1.0, 'surf_a1': 1.0}

        Returns
        -------
        jac_dict: dictionary
            A dictionary of the evaluated source Jacobian terms for the soot number density ('S_Ns') and soot mass fraction ('S_Ys').
            Keys include 'dSNs_dNs', 'dSNs_dYs', 'dSYs_dNs', 'dSYs_dYs'.
    """
    mult_dict = _get_default_mult_dict()
    if input_mult_dict is not None:
        mult_dict.update(input_mult_dict)

    area_fac_Ns = (1./3.)*np.power(soot_number_density, -2./3.) * np.power(soot_mass_fraction, 2./3.)
    area_fac_Ys = (2./3.)*np.power(soot_number_density, 1./3.) * np.power(soot_mass_fraction, -1./3.)

    coag_fac_Ns = (11./6.)*np.power(soot_number_density, 5./6.) * np.power(soot_mass_fraction, 1./6.)
    coag_fac_Ys = (1./6.)*np.power(soot_number_density, 11./6.) * np.power(soot_mass_fraction, -5./6.)

    coag_mole_rate = -mult_dict['coag'] * coag
    surf_c2h2_rate = mult_dict['surf_c2h2'] * sg
    surf_o2_rate = -mult_dict['surf_o2'] * ox_O2
    surf_oh_rate = -mult_dict['surf_oh'] * ox_OH
    surf_o_rate = -mult_dict['surf_o'] * ox_O

    dSNs_dNs = coag_mole_rate*coag_fac_Ns
    dSNs_dYs = coag_mole_rate*coag_fac_Ys
    dSYs_dNs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate+surf_o_rate)*area_fac_Ns
    dSYs_dYs = (surf_c2h2_rate+surf_o2_rate+surf_oh_rate+surf_o_rate)*area_fac_Ys

    jac_dict = {
        'dSNs_dNs': dSNs_dNs,
        'dSNs_dYs': dSNs_dYs,
        'dSYs_dNs': dSYs_dNs,
        'dSYs_dYs': dSYs_dYs,
    }
    return jac_dict


def get_soot_source_term_jacobian_evaluator(library, verbose=True):
    """
    Detect the soot model used in the library and return the corresponding Jacobian function

        Parameters
        ----------
        library: a spitfire.chemistry.library Library instance
            a library with soot source term coefficients, computed from running tabulate_soot_source_coefficients
        verbose: boolean
            whether or not to print the detected soot model on a library; default True

        Returns
        -------
        A function for evaluating soot source term Jacobian
    """
    soot_kinetics, _ = _detect_soot_model(library, verbose)

    if soot_kinetics=='aksit-moss':
        return jac_soot_src_aksit_moss
    elif soot_kinetics=='aksit-moss-benzene':
        return jac_soot_src_aksit_moss_benzene
    elif soot_kinetics=='pah-alpha':
        return jac_soot_src_pah_alpha
