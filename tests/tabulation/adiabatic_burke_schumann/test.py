import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from tests.tabulation.adiabatic_burke_schumann.rebless import run
from spitfire.chemistry.library import Library
from spitfire.chemistry.ctversion import check as cantera_version_check


if cantera_version_check('atleast', 2, 5, None):
    tol_args = {'atol': 1e-14} if cantera_version_check('pre', 2, 6, None) else {'atol': 1e-7, 'rtol': 3e-3}
    class Test(unittest.TestCase):
        def test(self):
            output_library = run()

            gold_file = abspath(join('tests',
                                     'tabulation',
                                     'adiabatic_burke_schumann',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], **tol_args))


    if __name__ == '__main__':
        unittest.main()
