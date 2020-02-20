import unittest
import numpy as np
from spitfire.chemistry.mechanism import ChemicalMechanismSpec
from spitfire.chemistry.tabulation import build_adiabatic_eq_library, Library, PostProcessor
from numpy.testing import assert_allclose
from os.path import join, abspath


class AdiabaticEquilibriumLibrary(unittest.TestCase):
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

        l1 = build_adiabatic_eq_library(flamelet_specs, quantities, verbose=False)

        file_name = abspath(join('spitfire_test',
                                 'unit',
                                 'tabulation',
                                 'gold_standards_test_' + 'adiabatic_eq_library',
                                 'library_gold.pkl'))
        l2 = Library.load_from_file(file_name)

        for prop in l1.props:
            self.assertIsNone(assert_allclose(l2[prop], l1[prop], atol=1.e-14))

    def test_with_post_processor(self):
        test_xml = abspath(join('spitfire_test', 'test_mechanisms', 'h2-burke.xml'))
        m = ChemicalMechanismSpec(cantera_xml=test_xml, group_name='h2-burke')
        pressure = 101325.
        air = m.stream(stp_air=True)
        air.TP = 1200., pressure
        fuel = m.stream('TPY', (300., pressure, 'H2:1'))

        flamelet_specs = {'mech_spec': m, 'pressure': pressure,
                          'oxy_stream': air, 'fuel_stream': fuel,
                          'grid_points': 34}

        quantities = ['enthalpy', 'temperature', 'mass fraction OH', 'mass fraction H2']

        post_proc_fxn = lambda t, y_H2: np.exp(-1000. / t) * y_H2

        class MyPostProc(PostProcessor):
            def __init__(self):
                super(MyPostProc, self).__init__(['temperature', 'mass fraction H2'])

            def evaluate(self, library):
                Y_H2 = library['mass fraction H2']
                T = library['temperature']
                library['post_proc'] = post_proc_fxn(T, Y_H2)
                return library

        l1 = build_adiabatic_eq_library(flamelet_specs, quantities, verbose=False,
                                        post_processors=[MyPostProc()])

        file_name = abspath(join('spitfire_test',
                                 'unit',
                                 'tabulation',
                                 'gold_standards_test_' + 'adiabatic_eq_library',
                                 'library_gold.pkl'))
        l2 = Library.load_from_file(file_name)

        for prop in l2.props:
            self.assertIsNone(assert_allclose(l2[prop], l1[prop], atol=1.e-14))

        post_proc = post_proc_fxn(l1['temperature'], l1['mass fraction H2'])
        self.assertIsNone(assert_allclose(post_proc, l1['post_proc'], atol=1.e-14))


if __name__ == '__main__':
    unittest.main()
