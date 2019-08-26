import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.reactors import HomogeneousReactor
        from os.path import abspath, join
        import matplotlib.pyplot as plt
        import numpy as np
        from time import perf_counter as timer
        import cantera

        cantera.suppress_thermo_warnings()

        T = 1000.
        p = 101325.

        mech_marker_dict = [('luo', 'bs'), ('williams', 'gD')]

        base_path = abspath(join('spitfire_test', 'test_mechanisms'))

        ntau = 80
        tau_vec = np.logspace(-7, 3, ntau)

        def make_mix_and_feed(spitfire_mech):
            air = spitfire_mech.stream(stp_air=True)
            fuel = spitfire_mech.stream('X', 'C2H4:1')

            mix = spitfire_mech.mix_for_equivalence_ratio(1., fuel, air)
            mix.TP = np.copy(T), np.copy(p)

            feed = spitfire_mech.mix_for_equivalence_ratio(1., fuel, air)
            feed.TP = T, p

            return mix, feed

        label_result_dict = dict()

        t0 = timer()
        for mech, marker in mech_marker_dict:
            b = ChemicalMechanismSpec(cantera_xml=base_path + '/ethylene-' + mech + '.xml',
                                      group_name='ethylene-' + mech)
            mix, feed = make_mix_and_feed(b)

            ns = mix.n_species
            label = mech + ' (' + str(ns) + ' sp.)'

            T_ignition_branch = np.zeros_like(tau_vec)
            T_extinction_branch = np.zeros_like(tau_vec)

            for idx, tau in enumerate(tau_vec):
                r = HomogeneousReactor(b, mix,
                                       configuration='isobaric',
                                       heat_transfer='adiabatic',
                                       mass_transfer='open',
                                       mixing_tau=tau,
                                       feed_temperature=feed.T,
                                       feed_mass_fractions=feed.Y)
                r.integrate_to_steady(steady_tolerance=1.e-8, transient_tolerance=1.e-10)
                T_ignition_branch[idx] = r.final_temperature
                mix.TPY = r.final_temperature, r.final_pressure, r.final_mass_fractions

            for idx, tau in enumerate(tau_vec[::-1]):
                r = HomogeneousReactor(b, mix,
                                       configuration='isobaric',
                                       heat_transfer='adiabatic',
                                       mass_transfer='open',
                                       mixing_tau=tau,
                                       feed_temperature=feed.T,
                                       feed_mass_fractions=feed.Y)
                r.integrate_to_steady(steady_tolerance=1.e-8, transient_tolerance=1.e-10)
                T_extinction_branch[idx] = r.final_temperature
                mix.TPY = r.final_temperature, r.final_pressure, r.final_mass_fractions

            label_result_dict[label] = (marker, np.copy(T_ignition_branch), np.copy(T_extinction_branch))

        incr = 1
        for label in label_result_dict:
            c = label_result_dict[label][0]
            Ti = label_result_dict[label][1]
            Te = label_result_dict[label][2]
            plt.semilogx(tau_vec, Ti, '-' + c, label=label, markevery=2 + incr, markerfacecolor='w')
            plt.semilogx(tau_vec[::-1], Te, '--' + c, markevery=2 + incr, markerfacecolor='k')
            incr += 1  # offsets the number of markers to better distinguish curves
        plt.grid()
        plt.xlabel('mixing time (s)')
        plt.ylabel('steady temperature (K)')
        plt.legend()


if __name__ == '__main__':
    unittest.main()
