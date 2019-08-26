"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.tabulation import build_adiabatic_slfm_library
import matplotlib.pyplot as plt
import numpy as np

m = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')

pressure = 101325.

air = m.stream(stp_air=True)
air.TP = 1400., pressure

fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), 0.2, air)
fuel.TP = 300., pressure

fig, axarr = plt.subplots(3, 1, sharex=True)
Tax = axarr[0]
hax = axarr[1]
Yax = axarr[2]

for include_flux, include_vcp, line_style, legend in \
        [(False, False, 'b-', 'classical'),
         (True, False, 'r--', 'w/h flux'),
         (True, True, 'g-.', 'w/h flux + var. cp')]:
    flamelet_specs = {'mech_spec': m,
                      'pressure': pressure,
                      'oxy_stream': air,
                      'fuel_stream': fuel,
                      'grid_points': 34,
                      'include_enthalpy_flux': include_flux,
                      'include_variable_cp': include_vcp}

    library = build_adiabatic_slfm_library(flamelet_specs,
                                           ['temperature', 'enthalpy', 'mass fraction OH'],
                                           diss_rate_ref='stoichiometric',
                                           diss_rate_values=np.logspace(0, 4, 4),
                                           verbose=False)

    x_values = library.dim('dissipation_rate_stoich').values
    z = library.dim('mixture_fraction').values

    for ix, (chi_st, marker) in enumerate(zip(x_values, ['s', 'o', 'd', '^'])):
        Tax.plot(z, library['temperature'][:, ix], line_style,
                 marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
        hax.plot(z, library['enthalpy'][:, ix] / 1.e6, line_style,
                 marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
        Yax.plot(z, library['mass fraction OH'][:, ix], line_style,
                 marker=marker, markevery=4, markersize=5, markerfacecolor='w', label=legend)
        Tmax = np.max(library['temperature'][:, ix])
        YOHmax = np.max(library['mass fraction OH'][:, ix])
        print(f'{legend:20} ' +
              f'X_st ={chi_st:10.1e} 1/s, ' +
              f'peak T = {Tmax:10.2f} K, ' +
              f'peak Y_OH = {YOHmax:10.2e}')
    print()

Yax.set_xlabel('mixture fraction')
Tax.set_ylabel('T (K)')
hax.set_ylabel('h (MJ/kg)')
Yax.set_ylabel('Y OH')
for ax in axarr:
    ax.grid(True)
Tax.legend(loc='best')
plt.show()
