from spitfire import ChemicalMechanismSpec, get_ct_solution_array, Library
import numpy as np
from cantera import gas_constant as Ru
from cantera import one_atm
from cantera import stefan_boltzmann as sigma


def tabulate_gas_radiation_model(mechanism, output_library, modelname='Barlow_GreyGas'):
    """Adds gas radiation properties to a library

        Parameters
        ----------
        mechanism: spitfire.chemistry.mechanism ChemicalMechanismSpec instance
            the mechanism
        output_library: a spitfire.chemistry.library Library instance
            a library with T, Y, and either density or pressure (pressure is used if both are present)
        modelname: str
            the gas radiation model. Supported models include: 'Barlow_GreyGas' (default)

        Returns
        -------
        output_library: a spitfire.chemistry.library Library instance
            The library with the following fields added

            - 'gas_abs_coeff': absorption coefficient
            - 'gas_rad_source': radiation source
    """
    if not isinstance(mechanism, ChemicalMechanismSpec):
        raise TypeError('Input argument "mechanism" must be a ChemicalMechanismSpec')
    if not isinstance(output_library, Library):
        raise TypeError('Input argument "output_library" must be a Library')
    ctsol, lib_shape = get_ct_solution_array(mechanism, output_library)

    if modelname == 'Barlow_GreyGas':
        ch4 = 'CH4'
        h2o = 'H2O'
        co = 'CO'
        co2 = 'CO2'

        T = ctsol.T
        rho = ctsol.density
        Tm1 = 1.e3 / T
        T2 = T * T

        absorption_fits = dict()
        absorption_fits[ch4] = 6.6334 - 0.0035686 * T + 1.6682e-08 * T2 + 2.5611e-10 * T2 * T - 2.6558e-14 * T2 * T2

        c0 = -0.23093
        c1 = -1.12390
        c2 = 9.41530
        c3 = -2.99880
        c4 = 0.51382
        c5 = -1.86840E-05
        absorption_fits[h2o] = c0 + Tm1 * (c1 + Tm1 * (c2 + Tm1 * (c3 + Tm1 * (c4 + Tm1 * c5))))

        c0 = 18.741
        c1 = -121.310
        c2 = 273.500
        c3 = -194.050
        c4 = 56.310
        c5 = -5.8169
        absorption_fits[co2] = c0 + Tm1 * (c1 + Tm1 * (c2 + Tm1 * (c3 + Tm1 * (c4 + Tm1 * c5))))

        c0 = 10.09
        c1 = -0.01183
        c2 = 4.7753e-6
        c3 = -5.87209e-10
        c4 = -2.5334e-14
        absorption_fits[co] = c0 + T * (c1 + T * (c2 + T * (c3 + T * c4)))
        c0 = 4.7869
        c1 = -0.06953
        c2 = 2.95775e-4
        c3 = -4.25732e-7
        c4 = 2.02894e-10
        absorption_fits[co][T < 750.] = c0 + T[T < 750.] * (c1 + T[T < 750.] * (c2 + T[T < 750.] * (c3 + T[T < 750.] * c4)))

        gas_abs_sum = np.zeros_like(T)
        for spec in absorption_fits:
            if spec in ctsol.species_names:
                idx = mechanism.species_index(spec)
                yi = ctsol.Y[:, idx]
                Mi = mechanism.molecular_weight(spec)
                gas_abs_sum += absorption_fits[spec] * yi / Mi

        alpha = (Ru / one_atm * rho * T * gas_abs_sum).reshape(lib_shape)
        output_library['gas_abs_coeff'] = alpha
        output_library['gas_rad_source'] = alpha * sigma / np.pi * np.power(output_library['temperature'], 4)
    else:
        raise ValueError("Supported gas radiation models include: Barlow_GreyGas.")
    return output_library
