"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

import cantera as ct
import numpy as np
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet import Flamelet
from spitfire.chemistry.tabulation import Dimension, Library
import pickle
from time import perf_counter


def get_fuel_stream(coal_fuels, alpha, mechanism, pressure):
    """
    Fuel streams representative of coal combustion, spanning char to volatiles.
    Personal communication with Josh McConnell, University of Utah, 2018.
    """

    volatiles = mechanism.stream('TPY', (coal_fuels['volatiles']['T'], pressure, coal_fuels['volatiles']['Y']))
    char = mechanism.stream('TPY', (coal_fuels['char']['T'], pressure, coal_fuels['char']['Y']))

    return mechanism.mix_streams([(volatiles, alpha), (char, 1. - alpha)], 'mass', 'HP')


mechanism = ChemicalMechanismSpec(cantera_xml='mechanisms/methane-gri30.xml', group_name='methane-gri30')
gas = mechanism.gas

pressure = ct.one_atm

oxy = mechanism.stream(stp_air=True)
oxy.TP = 350., pressure

particle_temperature = 350.

coal_fuels = pickle.load(open('coalflamelet_bcs.pkl', 'rb'))

alpha_vec = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
chist_vec = np.logspace(-1., 3., 6)
h_vec = np.hstack([0., np.logspace(-3, 1, 11)])

npts_interior = 128

outstr = 'alpha={:5.2f}, chi_st={:7.2e}, conv_coeff={:7.2e} complete in {:5.2f} s, total time={:6.2f} s'

base_specs = {'mech_spec': mechanism,
              'pressure': pressure,
              'oxy_stream': oxy,
              'grid_points': npts_interior + 2,
              'grid_type': 'uniform',
              'heat_transfer': 'nonadiabatic',
              'convection_temperature': particle_temperature,
              'radiation_temperature': particle_temperature,
              'radiative_emissivity': 0.}

quantities = ['temperature', 'enthalpy', 'mass fraction C2H2']

base_specs0 = dict(base_specs)
base_specs0.update({'fuel_stream': get_fuel_stream(coal_fuels, 0., mechanism, pressure),
                    'initial_condition': 'linear-TY',
                    'convection_coefficient': 0.})
f0 = Flamelet(**base_specs0)
zdim = Dimension('mixture_fraction', f0.mixfrac_grid)
xdim = Dimension('dissipation_rate_stoich', chist_vec)
hdim = Dimension('heat_transfer_coefficient', h_vec)
adim = Dimension('alpha', alpha_vec)

l = Library(adim, hdim, xdim, zdim)

values_dict = dict({q: l.get_empty_dataset() for q in quantities})

cput0 = perf_counter()
for ia, alpha in enumerate(alpha_vec):
    base_specs.update({'fuel_stream': get_fuel_stream(coal_fuels, alpha, mechanism, pressure)})

    for ichi, chist in enumerate(chist_vec):
        base_specs.update({'stoich_dissipation_rate': chist})

        for ih, h in enumerate(h_vec):
            base_specs.update({'convection_coefficient': h})
            base_specs.update({'initial_condition': 'equilibrium' if ichi == 0 and ih == 0 else f.final_interior_state})
            f = Flamelet(**base_specs)

            cput1 = perf_counter()
            f.compute_steady_state()
            dcput = perf_counter() - cput1
            print(outstr.format(alpha, chist, h, dcput, perf_counter() - cput0))

            data_dict = f.process_quantities_on_state(f.final_state, quantities)

            for quantity in quantities:
                values_dict[quantity][ia, ih, ichi, :] = data_dict[quantity].ravel()

for q in quantities:
    l[q] = values_dict[q]

l.save_to_file('coal_library.pkl')

print('full run complete in', perf_counter() - cput0, 's')
