import unittest
from os.path import join, abspath
import os

from numpy.testing import assert_allclose

from tests.tabulation.adiabatic_equilibrium.rebless import run
from spitfire.chemistry.library import Library
import spitfire.chemistry.analysis as sca
from spitfire.chemistry.ctversion import check as cantera_version_check


if cantera_version_check('atleast', 2, 5, None):
    tol_args = {'atol': 1e-14} if cantera_version_check('pre', 2, 6, None) else {'atol': 1e-7, 'rtol': 3e-3}
    class Test(unittest.TestCase):
        def test(self):
            output_library = run()

            gold_file = abspath(join('tests',
                                     'tabulation',
                                     'adiabatic_equilibrium',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], **tol_args))

            if os.path.isfile('testlib.pkl'):
                os.remove('testlib.pkl')
            output_library.save_to_file('testlib.pkl')
            l = Library.load_from_file('testlib.pkl')
            m = l.extra_attributes['mech_spec']
            l = sca.compute_specific_enthalpy(m, l)
            l = sca.compute_isochoric_specific_heat(m, l)
            l = sca.compute_isobaric_specific_heat(m, l)
            l = sca.compute_density(m, l)
            l = sca.compute_pressure(m, l)
            l = sca.compute_viscosity(m, l)
            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], l[prop], **tol_args))
            os.remove('testlib.pkl')


    if __name__ == '__main__':
        unittest.main()
