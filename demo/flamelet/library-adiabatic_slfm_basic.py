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
air.TP = 1200., pressure

fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), 0.2, air)
fuel.TP = 300., pressure

flamelet_specs = {'mech_spec': m,
                  'pressure': pressure,
                  'oxy_stream': air,
                  'fuel_stream': fuel,
                  'grid_points': 34}

quantities = ['temperature', 'density', 'mass fraction OH', 'viscosity']

library_coarse = build_adiabatic_slfm_library(flamelet_specs, quantities,
                                              diss_rate_ref='maximum', diss_rate_values=np.logspace(-1, 4, 6))

library_fine = build_adiabatic_slfm_library(flamelet_specs, quantities,
                                            diss_rate_ref='maximum', diss_rate_values=np.logspace(-1, 4, 40))

chi_list_coarse = library_coarse.dim('dissipation_rate_stoich').values
chi_list_fine = library_fine.dim('dissipation_rate_stoich').values
z = library_fine.dim('mixture_fraction').values

fig, axarray = plt.subplots(4, 1, sharex=True)
for i, chi_max in enumerate(chi_list_coarse):
    legend = '$\chi_{{max}}$={:.0e} Hz'.format(chi_max)
    axarray[0].plot(z, library_coarse['temperature'][:, i], label=legend)
    axarray[1].plot(z, library_coarse['density'][:, i])
    axarray[2].plot(z, library_coarse['mass fraction OH'][:, i])
    axarray[3].plot(z, library_coarse['viscosity'][:, i] * 1.e6)

axarray[0].set_ylabel('T (K)')
axarray[1].set_ylabel('density (kg/m3)')
axarray[2].set_ylabel('mass fraction OH')
axarray[3].set_ylabel('viscosity (uPa s)')
for ax in axarray:
    ax.grid(True)
axarray[0].legend(loc='best', fontsize=8)
axarray[-1].set_xlabel('mixture fraction')

fig2, axarray2 = plt.subplots(2, 2)
for i, quantity in enumerate(quantities):
    irow, icol = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}[i]
    q2d = np.swapaxes(library_fine[quantity], 0, 1)
    if quantity == 'viscosity':
        q2d *= 1.e6
    cf = axarray2[irow, icol].contourf(z, chi_list_fine, q2d)
    axarray2[irow, icol].set_yscale('log')
    axarray2[irow, icol].set_xlabel('mixture fraction')
    axarray2[irow, icol].set_ylabel('$\chi_{{max}}$')
    axarray2[irow, icol].set_title(quantity)
    plt.colorbar(cf, ax=axarray2[irow, icol], cmap=plt.get_cmap('inferno'))
for icol in range(2):
    axarray2[0, icol].set_xlabel('')
    axarray2[0, icol].set_xticks([])
for irow in range(2):
    axarray2[irow, 1].set_ylabel('')
    axarray2[irow, 1].set_yticks([])
plt.show()
