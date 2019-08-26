import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.reactors import HomogeneousReactor
        import matplotlib.pyplot as plt

        from os.path import abspath, join
        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        mechanism = ChemicalMechanismSpec(cantera_xml=xml,
                                          group_name='h2-burke')

        air = mechanism.stream(stp_air=True)
        fuel = mechanism.stream('X', 'H2:1')

        mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
        mix.TP = 1200., 101325.

        reactor_dict = {'cp, adiabatic': HomogeneousReactor(mechanism, mix, 'isobaric', 'adiabatic', 'closed'),
                        'cp, isothermal': HomogeneousReactor(mechanism, mix, 'isobaric', 'isothermal', 'closed'),
                        'cv, adiabatic': HomogeneousReactor(mechanism, mix, 'isochoric', 'adiabatic', 'closed'),
                        'cv, isothermal': HomogeneousReactor(mechanism, mix, 'isochoric', 'isothermal', 'closed')}

        for r in reactor_dict.values():
            r.insitu_process_quantity(['temperature', 'mass fractions'])
            r.integrate_to_steady()

        plt.subplot(211)
        for l in reactor_dict:
            r = reactor_dict[l]
            plt.semilogx(r.solution_times * 1.e6, r.trajectory_data('temperature'), label=l)
        plt.ylabel('T (K)')
        plt.legend()
        plt.grid()

        plt.subplot(212)
        for l in reactor_dict:
            r = reactor_dict[l]
            plt.semilogx(r.solution_times * 1.e6, r.trajectory_data('mass fraction H'), label=l)
        plt.ylabel('mass fraction H')

        plt.xlabel('t (us)')
        plt.legend()
        plt.grid()


if __name__ == '__main__':
    unittest.main()
