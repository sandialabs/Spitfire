from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt

mechanism = Mechanism('h2-burke.xml', 'h2-burke')

air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')

mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
mix.TP = 1200., 101325.

feed = mechanism.copy_stream(mix)

shape_list = HomogeneousReactor.get_supported_reactor_shapes()
shape_reactor_dict = dict()
solutions_dict = dict()

for shape in shape_list:
    shape_reactor_dict[shape] = HomogeneousReactor(mechanism, mix,
                                                   configuration='isobaric',
                                                   heat_transfer='diathermal',
                                                   mass_transfer='open',
                                                   mixing_tau=1.e-3,
                                                   feed_temperature=feed.T,
                                                   feed_mass_fractions=feed.Y,
                                                   convection_coefficient=100.0,
                                                   convection_temperature=300.,
                                                   radiative_emissivity=1.0,
                                                   radiation_temperature=300.,
                                                   shape_dimension_dict={'shape': shape, 'volume': 1.e-9})

for shape in shape_reactor_dict:
    solutions_dict[shape] = shape_reactor_dict[shape].integrate_to_steady()

for shape, r in shape_reactor_dict.items():
    plt.semilogx(solutions_dict[shape].time_values * 1.e6, solutions_dict[shape]['temperature'], label=shape)
plt.ylabel('T (K)')
plt.xlabel('t (us)')
plt.legend()
plt.grid()

plt.show()
