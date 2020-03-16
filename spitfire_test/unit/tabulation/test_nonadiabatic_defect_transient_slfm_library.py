import unittest
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.tabulation import build_nonadiabatic_defect_transient_slfm_library
from spitfire.chemistry.library import Library
from numpy.testing import assert_allclose
from os.path import join, abspath
import numpy as np


class NonadiabaticDefectSLFMLibrary(unittest.TestCase):
    def test(self):
        test_xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        m = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
        pressure = 101325.
        air = m.stream(stp_air=True)
        air.TP = 300., pressure
        fuel = m.stream('TPY', (300., pressure, 'H2:1'))

        flamelet_specs = {'mech_spec': m, 'pressure': pressure,
                          'oxy_stream': air, 'fuel_stream': fuel,
                          'grid_points': 64}

        l1 = build_nonadiabatic_defect_transient_slfm_library(flamelet_specs, verbose=False,
                                                              diss_rate_values=np.logspace(0, 1, 4))

        file_name = abspath(join('spitfire_test',
                                 'unit',
                                 'tabulation',
                                 'gold_standards_test_nonadiabatic_defect_transient_slfm_library',
                                 'library_gold.pkl'))
        l2 = Library.load_from_file(file_name)

        for prop in l1.props:
            self.assertIsNone(assert_allclose(l2[prop], l1[prop], rtol=1.e-4, atol=1.e-4))


if __name__ == '__main__':
    unittest.main()
