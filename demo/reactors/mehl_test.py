"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import linspace, zeros_like
from cantera import one_atm as atm
from time import perf_counter

b = ChemicalMechanismSpec(cantera_xml='/sierra/dev/mahanse/ChemicalMechanisms/octane-mehl/octane-mehl.xml',
                          group_name='octane-mehl')

air = b.stream(stp_air=True)
h2 = b.stream('X', 'IC8H18:1, CH3OCH3:1')
phi = 1.0
blend = b.mix_for_equivalence_ratio(phi, h2, air)

temperature_list = linspace(600., 1400., 20)
pressure_list = [10. * atm, 20. * atm, 100. * atm]
markers_list = ['o', 's', '^']

t0 = perf_counter()
for pressure, marker in zip(pressure_list, markers_list):
    print('computing ignition delay profile for {:.1f} atm'.format(pressure / atm))
    tau_list = zeros_like(temperature_list)

    for idx, temperature in enumerate(temperature_list):
        print('  computing temperature {:3} of {:2}'.format(idx + 1, temperature_list.size))

        mix = b.copy_stream(blend)
        mix.TP = temperature, pressure
        r = HomogeneousReactor(b, mix,
                               'isobaric',
                               'adiabatic',
                               'closed')
        tau_list[idx] = r.compute_ignition_delay(maximum_steps_per_jacobian=10,
                                                 first_time_step=1.e-9)

    plt.semilogy(temperature_list, tau_list * 1.e6, '-' + marker, label='{:.1f} atm'.format(pressure / atm))

print('completed in', perf_counter() - t0, 's')
plt.xlabel('T (K)')
plt.ylabel('ignition delay (us)')
plt.legend()
plt.grid()
plt.show()
