import unittest
import pickle
from os.path import join, abspath
from cantera import Solution
from spitfire import ChemicalMechanismSpec, CanteraLoadError


class MechanismSpec(unittest.TestCase):
    def test_create_valid_mechanism_spec_from_xml(self):
        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            g = m.griffon
            x = m.mech_xml_path
            s = m.gas
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_create_valid_mechanism_spec_from_solution(self):
        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            sol = Solution(xml)
            m = ChemicalMechanismSpec.from_solution(sol)
            g = m.griffon
            x = m.mech_xml_path
            n = m.group_name
            s = m.gas
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_create_invalid_mechanism_spec(self):
        try:
            ChemicalMechanismSpec('does not exist', 'at all')
            self.assertTrue(False)
        except CanteraLoadError:
            self.assertTrue(True)

    def test_create_invalid_mechanism_spec_bad_group(self):
        xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
        try:
            ChemicalMechanismSpec(xml, 'bad group')
            self.assertTrue(False)
        except CanteraLoadError:
            self.assertTrue(True)

    def test_create_valid_streams(self):
        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
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
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            m.stream(stp_air=False)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_create_invalid_streams_no_property_values(self):
        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec(xml, 'h2-burke')
            m.stream('TPX')
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_mechanism_simple_api_methods(self):
        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            sol = Solution(xml)
            m = ChemicalMechanismSpec.from_solution(sol)
            self.assertEqual(m.n_species, sol.n_species, 'ChemicalMechanismSpec.n_species vs ct.Solution.n_species')
            self.assertEqual(m.n_reactions, sol.n_reactions,
                             'ChemicalMechanismSpec.n_reactions vs ct.Solution.n_reactions')
            for i in range(m.n_species):
                self.assertEqual(m.species_names[i], sol.species_names[i],
                                 'ChemicalMechanismSpec.species_name[i] vs ct.Solution.species_name[i]')
            for n in m.species_names:
                self.assertEqual(m.species_index(n), sol.species_index(n),
                                 'ChemicalMechanismSpec.species_index(name) vs ct.Solution.species_index(name)')

            for i, n in enumerate(m.species_names):
                self.assertEqual(m.species_index(n), i, 'species names and indices are consistent, index vs i')
                self.assertEqual(n, m.species_names[i], 'species names and indices are consistent, name vs n')
                self.assertEqual(m.molecular_weight(i), m.molecular_weight(n),
                                 'ChemicalMechanismSpec molecular_weight(name) vs molecular_weight(idx)')
        except:
            self.assertTrue(False)

        try:
            xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
            m = ChemicalMechanismSpec.from_solution(Solution(xml))
            m.molecular_weight(list())
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_mechanism_serialization(self):
        xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.xml'))
        sol = Solution(xml)
        m1 = ChemicalMechanismSpec.from_solution(sol)

        m1_pickle = pickle.dumps(m1)
        m2 = pickle.loads(m1_pickle)

        self.assertTrue(m1.mech_data['species'] == m2.mech_data['species'])
        self.assertTrue(m1.mech_data['reactions'] == m2.mech_data['reactions'])



if __name__ == '__main__':
    unittest.main()
