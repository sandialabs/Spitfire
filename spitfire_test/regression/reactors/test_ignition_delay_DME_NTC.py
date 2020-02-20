import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.reactors import HomogeneousReactor
        import matplotlib.pyplot as plt
        from numpy import linspace, zeros_like
        from cantera import one_atm

        from os.path import abspath, join
        xml = abspath(join('spitfire_test', 'test_mechanisms', 'dme-bhagatwala.xml'))
        b = ChemicalMechanismSpec(cantera_xml=xml,
                                  group_name='dme-bhagatwala')

        air = b.stream(stp_air=True)
        h2 = b.stream('X', 'CH3OCH3:1, CH4:1')
        phi = 1.0
        blend = b.mix_for_equivalence_ratio(phi, h2, air)

        temperature_list = linspace(600., 1800., 4)
        pressure_atm_list = [2.]
        markers_list = ['o', 's', '^', 'D', 'P', '*']

        for pressure, marker in zip(pressure_atm_list, markers_list):
            tau_list = zeros_like(temperature_list)

            for idx, temperature in enumerate(temperature_list):
                mix = b.copy_stream(blend)
                mix.TP = temperature, pressure * one_atm

                r = HomogeneousReactor(b, mix,
                                       'isobaric',
                                       'adiabatic',
                                       'closed')
                tau_list[idx] = r.compute_ignition_delay(first_time_step=1.e-9)

            plt.semilogy(1. / temperature_list, tau_list * 1.e6, '-' + marker, label='{:.1f} atm'.format(pressure))

        plt.xlabel('1/T (1/K)')
        plt.ylabel('ignition delay (us)')
        plt.legend()
        plt.grid()


if __name__ == '__main__':
    unittest.main()
