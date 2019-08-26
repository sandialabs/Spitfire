"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import linspace, zeros_like, array
from cantera import one_atm
from time import perf_counter as timer

b = ChemicalMechanismSpec(cantera_xml='mechanisms/dme-bhagatwala.xml', group_name='dme-bhagatwala')

air = b.stream(stp_air=True)
h2 = b.stream('X', 'CH3OCH3:1, CH4:1')
phi = 1.0
blend = b.mix_for_equivalence_ratio(phi, h2, air)

temperature_list = linspace(600., 1800., 40)
# pressure_atm_list = [4., 10., 20., 50., 100., 200.]
pressure_atm_list = [4.]
markers_list = ['o', 's', '^', 'D', 'P', '*']

t0 = timer()
for pressure, marker in zip(pressure_atm_list, markers_list):
    print('computing ignition delay profile for {:.1f} atm'.format(pressure))
    tau_list = zeros_like(temperature_list)

    for idx, temperature in enumerate(temperature_list):
        print('  computing temperature {:3} of {:2}'.format(idx + 1, temperature_list.size))

        mix = b.copy_stream(blend)
        mix.TP = temperature, pressure * one_atm

        r = HomogeneousReactor(b, mix,
                               'isobaric',
                               'adiabatic',
                               'closed')
        # tau_list[idx] = r.compute_ignition_delay(first_time_step=1.e-9)
        tau_list[idx] = r.compute_ignition_delay_direct_griffon()

    plt.semilogy(1. / temperature_list, tau_list * 1.e6, '-' + marker, label='{:.1f} atm'.format(pressure))

print('completed in', timer() - t0, 's')
plt.xlabel('1/T (1/K)')
plt.ylabel('ignition delay (us)')
plt.legend()
plt.grid()
plt.show()
