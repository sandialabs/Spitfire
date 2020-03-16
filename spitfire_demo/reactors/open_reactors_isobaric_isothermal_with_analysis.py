from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
from spitfire.chemistry.analysis import explosive_mode_analysis
import matplotlib.pyplot as plt
import numpy as np

mechanism = Mechanism('h2-burke.xml', 'h2-burke')

air = mechanism.stream(stp_air=True)
fuel = mechanism.stream('X', 'H2:1')

mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
mix.TP = 1200., 101325.

feed = mechanism.copy_stream(mix)

tau_list = [1.e-6, 5.e-6, 1.e-5, 5.e-5, 1.e-4, 5.e-4]
tau_reactor_dict = dict()

closed_reactor = HomogeneousReactor(mechanism, mix,
                                    configuration='isobaric',
                                    heat_transfer='isothermal',
                                    mass_transfer='closed')

output_closed = closed_reactor.integrate_to_steady()

output_closed = explosive_mode_analysis(mechanism,
                                        output_closed,
                                        configuration='isobaric',
                                        heat_transfer='isothermal',
                                        compute_explosion_indices=True,
                                        compute_participation_indices=True,
                                        include_secondary_mode=False)

plt.loglog(output_closed.time_values * 1.e6,
           output_closed['cema-lexp1'])
plt.ylabel('Explosive eigenvalue')
plt.xlabel('t (us)')
plt.show()

for name in ['T'] + mechanism.species_names[:-1]:
    ei = output_closed['cema-ei1 ' + name]
    if np.max(ei) > 0.1:
        plt.semilogx(output_closed.time_values * 1.e6, ei, label=name)
plt.ylabel('Explosion index')
plt.xlabel('t (us)')
plt.legend()
plt.show()

for i in range(mechanism.n_reactions):
    pi = output_closed['cema-pi1 ' + str(i)]
    if np.max(pi) > 0.2:
        plt.semilogx(output_closed.time_values * 1.e6, pi,
                     label=mechanism.gas.reaction_equation(i))
plt.ylabel('Participation index')
plt.xlabel('t (us)')
plt.legend()
plt.show()

for tau in tau_list:
    tau_reactor_dict[tau] = HomogeneousReactor(mechanism, mix,
                                               configuration='isobaric',
                                               heat_transfer='isothermal',
                                               mass_transfer='open',
                                               mixing_tau=tau,
                                               feed_temperature=feed.T,
                                               feed_mass_fractions=feed.Y)

output_dict = dict()
for tau in tau_reactor_dict:
    output_dict[tau] = tau_reactor_dict[tau].integrate_to_steady()

plt.loglog(output_closed.time_values * 1.e6,
           output_closed['mass fraction H'],
           '--', label='closed')
for tau in tau_reactor_dict:
    plt.loglog(output_dict[tau].time_values * 1.e6,
               output_dict[tau]['mass fraction H'], '-',
               label='open, $\\tau={:.0f}$ $\mu$s'.format(tau * 1.e6))
plt.ylabel('mass fraction H')
plt.xlabel('t (us)')
plt.legend()
plt.grid()

plt.show()
