from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
from numpy import sin as sin, pi as pi

mechanism = ChemicalMechanismSpec(cantera_xml='h2-burke.xml', group_name='h2-burke')

air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')

mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
mix.TP = 1000., 101325.

feed = mechanism.copy_stream(mix)
feed.TP = 1400., 101325.

external_temperature = 300.
convection_coefficient_fxn = lambda t: 20. + 18. * sin(2. * pi * 5. * t)

reactor = HomogeneousReactor(mechanism, mix,
                             configuration='isobaric',
                             heat_transfer='diathermal',
                             mass_transfer='open',
                             mixing_tau=8.e-3,
                             feed_temperature=feed.T,
                             feed_mass_fractions=feed.Y,
                             convection_temperature=external_temperature,
                             convection_coefficient=convection_coefficient_fxn,
                             radiation_temperature=external_temperature,
                             radiative_emissivity=0.0,
                             shape_dimension_dict={'shape': 'tetrahedron', 'char. length': 1.e-3})

output = reactor.integrate_to_time(0.4, write_log=True, log_rate=50)

times = output.time_values

plt.subplot(211)
plt.plot(times * 1.e3, output['temperature'], label='reactor')
plt.plot(times * 1.e3, times * 0. + feed.T, '--', label='feed')
plt.plot(times * 1.e3, times * 0. + external_temperature, '--', label='convection fluid')
plt.ylabel('T (K)')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(times * 1.e3, convection_coefficient_fxn(times))
plt.ylabel('convection coefficient (W/m2/s)')
plt.xlabel('t (ms)')
plt.grid()

plt.show()
