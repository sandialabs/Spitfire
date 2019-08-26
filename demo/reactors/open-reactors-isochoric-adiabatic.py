"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt

mechanism = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')

air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')

mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
mix.TP = 1200., 101325.

feed = mechanism.copy_stream(mix)

tau_list = [1.e-6, 5.e-6, 1.e-5, 5.e-5, 1.e-4, 5.e-4]
tau_reactor_dict = dict()

closed_reactor = HomogeneousReactor(mechanism, mix,
                                    configuration='isochoric',
                                    heat_transfer='adiabatic',
                                    mass_transfer='closed')
closed_reactor.insitu_process_quantity('temperature')

for tau in tau_list:
    tau_reactor_dict[tau] = HomogeneousReactor(mechanism, mix,
                                               configuration='isochoric',
                                               heat_transfer='adiabatic',
                                               mass_transfer='open',
                                               mixing_tau=tau,
                                               feed_temperature=feed.T,
                                               feed_mass_fractions=feed.Y,
                                               feed_density=feed.density)

closed_reactor.integrate_to_steady()

for r in tau_reactor_dict.values():
    r.insitu_process_quantity('temperature')
    r.integrate_to_steady()

plt.semilogx(closed_reactor.solution_times * 1.e6, closed_reactor.trajectory_data('temperature'), '--', label='closed')
for tau, r in tau_reactor_dict.items():
    plt.semilogx(r.solution_times * 1.e6, r.trajectory_data('temperature'), '-',
                 label='open, $\\tau={:.0f}$ $\mu$s'.format(tau * 1.e6))
plt.ylabel('T (K)')
plt.xlabel('t (us)')
plt.legend()
plt.grid()

plt.show()
