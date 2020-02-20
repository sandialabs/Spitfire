import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.reactors import HomogeneousReactor
        from os.path import abspath, join

        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        sm = ChemicalMechanismSpec(cantera_xml=xml, group_name='h2-burke')

        h2 = sm.stream('X', 'H2:1')
        air = sm.stream(stp_air=True)

        mix = sm.mix_for_equivalence_ratio(1.0, h2, air)
        mix.TP = 1200, 101325

        r = HomogeneousReactor(sm, mix,
                               configuration='isobaric',
                               heat_transfer='adiabatic',
                               mass_transfer='closed')

        r.insitu_process_quantity(['temperature', 'mass fractions', 'production rates'])
        r.insitu_process_cantera_method('cp_mass')
        r.insitu_process_cantera_method(label='cpm', method='cp_mass')
        r.insitu_process_cantera_method(label='qCB', method='net_rates_of_progress', index=0)
        r.insitu_process_cantera_method(label='cH', method='concentrations', index='H')
        r.insitu_process_cema()

        r.integrate_to_steady_after_ignition()

        t = r.solution_times * 1.e6  # scale to microseconds
        T = r.trajectory_data('temperature')
        yH = r.trajectory_data('mass fraction H')
        wH = r.trajectory_data('production rate H')
        qCB = r.trajectory_data('qCB')


if __name__ == '__main__':
    unittest.main()
