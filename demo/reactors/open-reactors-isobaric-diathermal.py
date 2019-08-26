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

size_list = [1.e-4, 3.e-4, 4.e-4, 5.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0]
size_reactor_dict = dict()

for size in size_list:
    size_reactor_dict[size] = HomogeneousReactor(mechanism, mix,
                                                 configuration='isobaric',
                                                 heat_transfer='diathermal',
                                                 mass_transfer='open',
                                                 mixing_tau=1.e-3,
                                                 feed_temperature=feed.T,
                                                 feed_mass_fractions=feed.Y,
                                                 convection_coefficient=1.0,
                                                 convection_temperature=300.,
                                                 radiative_emissivity=1.0,
                                                 radiation_temperature=300.,
                                                 shape_dimension_dict={'shape': 'cube', 'char. length': size})

for r in size_reactor_dict.values():
    r.insitu_process_quantity('temperature')
    r.integrate_to_steady()

for size, r in size_reactor_dict.items():
    size_str = '{:.0f} mm'.format(size * 1.e3) if size > 9.e-4 else '{:.0f} $\mu$m'.format(size * 1.e6)
    plt.semilogx(r.solution_times * 1.e6, r.trajectory_data('temperature'), label=size_str)
plt.ylabel('T (K)')
plt.xlabel('t (us)')
plt.legend()
plt.grid()

plt.show()
