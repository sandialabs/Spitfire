"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.flamelet import Flamelet
import matplotlib.pyplot as plt

m = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')

pressure = 101325.

air = m.stream(stp_air=True)
air.TP = 1400., pressure

zstoich = 0.1

fuel = m.mix_fuels_for_stoich_mixture_fraction(m.stream('X', 'H2:1'), m.stream('X', 'N2:1'), zstoich, air)
fuel.TP = 300., pressure

chi_max = 1.e3

npts_interior = 32

for ht, c in [('nonadiabatic', 'r'), ('adiabatic', 'b')]:
    ft = Flamelet(mech_spec=m,
                  pressure=pressure,
                  oxy_stream=air,
                  fuel_stream=fuel,
                  max_dissipation_rate=chi_max,
                  grid_points=npts_interior + 2,
                  grid_cluster_intensity=4.,
                  initial_condition='unreacted',
                  convection_coefficient=5.e4,
                  convection_temperature=300.,
                  radiation_temperature=300.,
                  radiative_emissivity=1.,
                  heat_transfer=ht)

    ft.insitu_process_quantity('temperature')
    ft.integrate_to_steady(log_rate=80, write_log=True)

    t = ft.solution_times
    z = ft.mixfrac_grid
    T = ft.trajectory_data('temperature')

    ntimes = t.size
    mid_idx = ntimes // 2
    mid_t = ft.solution_times[mid_idx] * 1.e6
    mid_idx2 = ntimes // 2 + ntimes // 4
    mid_t2 = ft.solution_times[mid_idx2] * 1.e6

    plt.plot(z, T[0, :], c + '--', label='t=0 us')
    plt.plot(z, T[mid_idx, :], c + '-.', label='t={:.0f} us'.format(mid_t))
    plt.plot(z, T[mid_idx2, :], c + ':', label='t={:.0f} us'.format(mid_t2))
    plt.plot(z, T[-1, :], c + '-', label='steady')

plt.title('red: nonadiabatic, blue: adiabatic')
plt.xlabel('Z')
plt.ylabel('T (K)')
plt.legend()
plt.grid()
plt.show()
