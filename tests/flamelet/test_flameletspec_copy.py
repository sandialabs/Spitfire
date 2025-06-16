import unittest
from os.path import join, abspath
from spitfire import ChemicalMechanismSpec, FlameletSpec


class Test(unittest.TestCase):
    def test_grid_copy(self):
        test_xml = abspath(join('tests', 'test_mechanisms', 'h2-burke.yaml'))
        mechanism = ChemicalMechanismSpec(cantera_input=test_xml, group_name='h2-burke')
        air = mechanism.stream(stp_air=True)
        fuel = mechanism.stream('X', 'H2:1')
        fuel.TP = 300, air.P
        flamelet_specs = {'mech_spec': mechanism,
                          'oxy_stream': air,
                          'fuel_stream': fuel,
                          'initial_condition': 'equilibrium',
                          'grid_points':21}
        fso = FlameletSpec(**flamelet_specs)
        if fso.grid is not None:
            f2 = fso.copy()
            if f2.grid is fso.grid:
                self.assertTrue(False)
            else:
                self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
