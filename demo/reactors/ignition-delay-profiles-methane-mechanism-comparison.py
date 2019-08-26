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
from time import perf_counter as timer

fuel = 'methane'
mechanism_marker_dict = {'gri30': 'o',
                         'kazakov21': 's',
                         'kazakov24': 'd',
                         'lu30': 'P'}


def make_blend(mech):
    spitfire_mech = ChemicalMechanismSpec(cantera_xml='mechanisms/' + fuel + '-' + mech + '.xml',
                                          group_name=fuel + '-' + mech)
    air = spitfire_mech.stream(stp_air=True)
    h2 = spitfire_mech.stream('X', 'CH4:1')
    phi = 1.0
    blend = spitfire_mech.mix_for_equivalence_ratio(phi, h2, air)
    return spitfire_mech, blend


temperature_list = linspace(800., 1400., 20)

t0 = timer()
for mech in mechanism_marker_dict:
    print('computing ignition delay profile for mechanism:', mech)
    marker = mechanism_marker_dict[mech]
    spitfire_mech, blend = make_blend(mech)
    tau_list = zeros_like(temperature_list)

    for idx, temperature in enumerate(temperature_list):
        print('  computing temperature {:3} of {:2}'.format(idx + 1, temperature_list.size))

        mix = spitfire_mech.copy_stream(blend)
        mix.TP = temperature, 101325.

        r = HomogeneousReactor(spitfire_mech, mix,
                               'isochoric',
                               'adiabatic',
                               'closed')
        tau_list[idx] = r.compute_ignition_delay(first_time_step=1.e-9)

    plt.semilogy(temperature_list, tau_list, '-' + marker, label=mech + ' (' + str(mix.n_species) + 'sp.)')

print('completed in', timer() - t0, 's')
plt.xlabel('T (K)')
plt.ylabel('ignition delay (s)')
plt.legend()
plt.grid()
plt.show()
