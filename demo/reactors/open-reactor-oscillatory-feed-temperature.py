"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import sin as sin, pi as pi

mechanism = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-burke.xml', group_name='h2-burke')

air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')

mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
mix.TP = 800., 101325.

feed = mechanism.copy_stream(mix)
feed_temperature_fxn = lambda t: 800. + 400. * sin(2. * pi * 10. * t)

reactor = HomogeneousReactor(mechanism, mix,
                             configuration='isobaric',
                             heat_transfer='adiabatic',
                             mass_transfer='open',
                             mixing_tau=1.e-5,
                             feed_temperature=feed_temperature_fxn,
                             feed_mass_fractions=feed.Y)

reactor.insitu_process_quantity('temperature')
reactor.integrate_to_time(0.2, transient_tolerance=1.e-10, write_log=True, log_rate=200)

times = reactor.solution_times

plt.plot(times * 1.e3, reactor.trajectory_data('temperature'), '-', label='reactor')
plt.plot(times * 1.e3, feed_temperature_fxn(times), '--', label='feed')

plt.ylabel('T (K)')
plt.xlabel('t (ms)')
plt.legend()
plt.grid()

plt.show()
