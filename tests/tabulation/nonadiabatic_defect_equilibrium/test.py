import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from tests.tabulation.nonadiabatic_defect_equilibrium.rebless import run
from spitfire.chemistry.library import Library
from spitfire.chemistry.ctversion import check as cantera_version_check


if cantera_version_check('atleast', 2, 5, None):
    tol_args = {'atol': 1e-14} if cantera_version_check('pre', 2, 6, None) else {'atol': 1e-7, 'rtol': 3e-3}
    class Test(unittest.TestCase):
        def test(self):
            output_library = run()

            gold_file = abspath(join('tests',
                                     'tabulation',
                                     'nonadiabatic_defect_equilibrium',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], **tol_args))

        def test_stoich_heatloss_multiplier(self):
            import numpy as np
            from spitfire import ChemicalMechanismSpec, FlameletSpec, build_nonadiabatic_defect_eq_library
            from cantera._utils import CanteraError
            yamlfile = abspath(join('tests','test_mechanisms','heptane-liu-hewson-chen-pitsch-highT.yaml'))
            mech = ChemicalMechanismSpec(yamlfile, 'gas')
            pressure = 101325.
            air = mech.stream(stp_air=True)
            fuel = mech.stream('TPX', (485., pressure, 'NXC7H16:1'))
            fuel.equilibrate("HP")

            flamelet_specs = FlameletSpec(mech_spec=mech,
                                          initial_condition='equilibrium',
                                          oxy_stream=air,
                                          fuel_stream=fuel,
                                          grid_points=128)
            try: # unity multiplier
                l_unity = build_nonadiabatic_defect_eq_library(flamelet_specs, n_defect_st=5, stoich_heatloss_multiplier=1., verbose=False)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

            try: # non-unity multiplier, okay
                mult = 1.05
                l_new = build_nonadiabatic_defect_eq_library(flamelet_specs, n_defect_st=5, stoich_heatloss_multiplier=mult, verbose=False)
                self.assertTrue(np.abs(l_new.enthalpy_defect_stoich_values[-1]/l_unity.enthalpy_defect_stoich_values[-1] - mult) <= 1.e-12)
                self.assertTrue(True)
            except ValueError:
                self.assertTrue(False)

            try: # cantera solver failure
                l_fail = build_nonadiabatic_defect_eq_library(flamelet_specs, n_defect_st=5, stoich_heatloss_multiplier=1.3, verbose=False)
                self.assertTrue(False)
            except CanteraError:
                self.assertTrue(True)


    if __name__ == '__main__':
        unittest.main()
