"""
Copyright (c) 2018-2019 Michael Alan Hansen - All Rights Reserved
You may use, distribute and modify this code under the terms of the MIT license.

You should have received a copy of the MIT license with this file.
If not, please write to mahanse@sandia.gov or mike.hansen@chemeng.utah.edu
"""

from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.reactors import HomogeneousReactor
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter as timer

T = 800.
p = 2. * 101325.

fuel_name = 'hydrogen'
mech_marker_dict = [('oconaire', 'ro'), ('li', 'bs'), ('burke', 'cD'), ('sun', 'g+')]

ntau = 80
tau_vec = np.logspace(-7, 3, ntau)


def make_mix_and_feed(spitfire_mech):
    air = spitfire_mech.stream(stp_air=True)
    fuel = spitfire_mech.stream('X', 'H2:1')

    mix = spitfire_mech.mix_for_equivalence_ratio(1., fuel, air)
    mix.TP = np.copy(T), np.copy(p)

    feed = spitfire_mech.mix_for_equivalence_ratio(1., fuel, air)
    feed.TP = T, p

    return mix, feed


label_result_dict = dict()

t0 = timer()
for mech, marker in mech_marker_dict:
    b = ChemicalMechanismSpec(cantera_xml='mechanisms/h2-' + mech + '.xml', group_name='h2-' + mech)

    mix, feed = make_mix_and_feed(b)

    ns = mix.n_species
    label = mech + ' (' + str(ns) + ' sp)'

    T_ignition_branch = np.zeros_like(tau_vec)
    T_extinction_branch = np.zeros_like(tau_vec)

    status_str = lambda branch, tau, T, idx: ': {:10} branch : ' \
                                             'tau = {:6.1e} s, ' \
                                             'T = {:7.3f} K, ' \
                                             '{:>4} / {:<4}'.format(branch, tau, T, idx + 1, ntau)

    for idx, tau in enumerate(tau_vec):
        r = HomogeneousReactor(b, mix,
                               configuration='isobaric',
                               heat_transfer='adiabatic',
                               mass_transfer='open',
                               mixing_tau=tau,
                               feed_temperature=feed.T,
                               feed_mass_fractions=feed.Y)
        r.integrate_to_steady_direct_griffon(steady_tolerance=1.e-8, transient_tolerance=1.e-14)
        T_ignition_branch[idx] = r.final_temperature
        mix.TPY = r.final_temperature, r.final_pressure, r.final_mass_fractions
        print(mech, status_str('Ignition', tau, r.final_temperature, idx))

    for idx, tau in enumerate(tau_vec[::-1]):
        r = HomogeneousReactor(b, mix,
                               configuration='isobaric',
                               heat_transfer='adiabatic',
                               mass_transfer='open',
                               mixing_tau=tau,
                               feed_temperature=feed.T,
                               feed_mass_fractions=feed.Y)
        r.integrate_to_steady_direct_griffon(steady_tolerance=1.e-8, transient_tolerance=1.e-14)
        T_extinction_branch[idx] = r.final_temperature
        mix.TPY = r.final_temperature, r.final_pressure, r.final_mass_fractions
        print(mech, status_str('Extinction', tau, r.final_temperature, idx))

    label_result_dict[label] = (marker, np.copy(T_ignition_branch), np.copy(T_extinction_branch))

print('all mechanisms completed in ', timer() - t0, 's')

incr = 1
for label in label_result_dict:
    c = label_result_dict[label][0]
    Ti = label_result_dict[label][1]
    Te = label_result_dict[label][2]
    plt.semilogx(tau_vec, Ti, '-' + c, label=label, markevery=2 + incr, markerfacecolor='w')
    plt.semilogx(tau_vec[::-1], Te, '--' + c, markevery=2 + incr, markerfacecolor='k')
    incr += 1  # simply to offset the number of markers to better distinguish curves
plt.grid()
plt.xlabel('mixing time (s)')
plt.ylabel('steady temperature (K)')
plt.legend()
plt.show()
