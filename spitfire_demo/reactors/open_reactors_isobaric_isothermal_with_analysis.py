from spitfire.chemistry.mechanism import ChemicalMechanismSpec as Mechanism
from spitfire.chemistry.reactors import HomogeneousReactor
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


def setup_insitu_processing(reactor):
    reactor.insitu_process_quantity('mass fractions')
    reactor.insitu_process_cantera_method('cp', 'cp_mass')
    reactor.insitu_process_cantera_method('rxn0', 'net_rates_of_progress', 0)
    reactor.insitu_process_cantera_method('wH', 'net_production_rates', 'H')
    reactor.insitu_process_cema(secondary_mode=True, explosion_indices=True, participation_indices=True)


closed_reactor = HomogeneousReactor(mechanism, mix,
                                    configuration='isobaric',
                                    heat_transfer='isothermal',
                                    mass_transfer='closed')
setup_insitu_processing(closed_reactor)

for tau in tau_list:
    tau_reactor_dict[tau] = HomogeneousReactor(mechanism, mix,
                                               configuration='isobaric',
                                               heat_transfer='isothermal',
                                               mass_transfer='open',
                                               mixing_tau=tau,
                                               feed_temperature=feed.T,
                                               feed_mass_fractions=feed.Y)

closed_reactor.integrate_to_steady()

for r in tau_reactor_dict.values():
    setup_insitu_processing(r)
    r.integrate_to_steady()

for name in ['T'] + closed_reactor._gas.species_names[:-1]:
    ei = closed_reactor.trajectory_data('cema-ei1 ' + name)
    if np.max(ei) > 0.1:
        plt.semilogx(closed_reactor.solution_times * 1.e6, ei,
                     label=name)
plt.ylabel('Explosion index')
plt.xlabel('t (us)')
plt.legend()
plt.show()

for i in range(closed_reactor._n_reactions):
    pi = closed_reactor.trajectory_data('cema-pi1 ' + str(i))
    if np.max(pi) > 0.2:
        plt.semilogx(closed_reactor.solution_times * 1.e6, pi,
                     label=closed_reactor._gas.reaction_equation(i))
plt.ylabel('Participation index')
plt.xlabel('t (us)')
plt.legend()
plt.show()

plt.loglog(closed_reactor.solution_times * 1.e6,
           closed_reactor.trajectory_data('mass fraction H'),
           '--', label='closed')
for tau, r in tau_reactor_dict.items():
    plt.loglog(r.solution_times * 1.e6, r.trajectory_data('mass fraction H'), '-',
               label='open, $\\tau={:.0f}$ $\mu$s'.format(tau * 1.e6))
plt.ylabel('Y H')
plt.xlabel('t (us)')
plt.legend()
plt.grid()

plt.show()
