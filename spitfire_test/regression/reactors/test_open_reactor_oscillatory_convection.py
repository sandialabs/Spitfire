import unittest


class Test(unittest.TestCase):
    def test(self):
        from spitfire.chemistry.mechanism import ChemicalMechanismSpec
        from spitfire.chemistry.reactors import HomogeneousReactor
        import matplotlib.pyplot as plt
        from numpy import sin as sin, pi as pi

        from os.path import abspath, join
        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        mechanism = ChemicalMechanismSpec(cantera_xml=xml,
                                          group_name='h2-burke')

        air = mechanism.stream(stp_air=True)
        fuel = mechanism.stream('X', 'H2:1')

        mix = mechanism.mix_for_equivalence_ratio(1.0, fuel, air)
        mix.TP = 1000., 101325.

        feed = mechanism.copy_stream(mix)
        feed.TP = 1400., 101325.

        external_temperature = 300.

        reactor = HomogeneousReactor(mechanism, mix,
                                     configuration='isobaric',
                                     heat_transfer='diathermal',
                                     mass_transfer='open',
                                     mixing_tau=8.e-3,
                                     feed_temperature=feed.T,
                                     feed_mass_fractions=feed.Y,
                                     convection_temperature=external_temperature,
                                     convection_coefficient=lambda t: 20. + 18. * sin(2. * pi * 5. * t),
                                     radiation_temperature=external_temperature,
                                     radiative_emissivity=0.0,
                                     shape_dimension_dict={'shape': 'tetrahedron', 'char. length': 1.e-3})

        reactor.insitu_process_quantity('temperature')
        reactor.integrate_to_time(0.4, write_log=False, log_rate=50)


if __name__ == '__main__':
    unittest.main()
