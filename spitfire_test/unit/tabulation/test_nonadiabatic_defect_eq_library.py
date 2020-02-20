import unittest
import numpy as np
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.tabulation import build_nonadiabatic_defect_eq_library, Library, PostProcessor
from numpy.testing import assert_allclose
from os.path import join, abspath


class NonadiabaticDefectEquilibriumLibrary(unittest.TestCase):
    def test_basic(self):
        test_xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        m = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
        pressure = 101325.
        air = m.stream(stp_air=True)
        air.TP = 1200., pressure
        fuel = m.stream('TPY', (300., pressure, 'H2:1'))

        flamelet_specs = {'mech_spec': m, 'pressure': pressure,
                          'oxy_stream': air, 'fuel_stream': fuel,
                          'grid_points': 34}

        quantities = ['enthalpy', 'temperature', 'mass fraction OH']

        l1 = build_nonadiabatic_defect_eq_library(flamelet_specs, quantities, verbose=False)

        file_name = abspath(join('spitfire_test',
                                 'unit',
                                 'tabulation',
                                 'gold_standards_test_' + 'nonadiabatic_defect_eq_library',
                                 'library_gold.pkl'))
        l2 = Library.load_from_file(file_name)

        for prop in l1.props:
            self.assertIsNone(assert_allclose(l2[prop], l1[prop], atol=1.e-14))


if __name__ == '__main__':
    unittest.main()
