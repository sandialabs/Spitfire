import unittest
from os.path import join, abspath
from spitfire.chemistry.mechanism import ChemicalMechanismSpec


class MechanismSpec(unittest.TestCase):
    def test_create_valid_mechanism_spec(self):
        try:
            xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            g = m.griffon
            x = m.mech_xml_path
            s = m.gas
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_create_invalid_mechanism_spec(self):
        try:
            ChemicalMechanismSpec('does not exist', 'at all')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_create_invalid_mechanism_spec_bad_group(self):
        xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        try:
            ChemicalMechanismSpec(xml, 'bad group')
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_create_valid_streams(self):
        try:
            xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            air = m.stream(stp_air=True)
            h2o2 = m.stream('TPX', (300., 101325., 'H2:1, O2:1'))
            air_copy = m.copy_stream(air)
            mix = m.mix_streams([(h2o2, 1.), (air, 1.)], 'mass', 'HP')
            mix = m.mix_streams([(h2o2, 1.), (air, 1.)], 'mole', 'TP')
            mix = m.mix_streams([(h2o2, 1.), (air, 1.)], 'mass', 'UV')
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_create_invalid_stpair_streams(self):
        try:
            xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            m.stream(stp_air=False)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_create_invalid_streams_no_property_values(self):
        try:
            xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            m.stream('TPX')
            self.assertTrue(False)
        except:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
