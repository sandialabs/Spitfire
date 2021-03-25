import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from tests.tabulation.adiabatic_slfm.rebless import run
from spitfire.chemistry.library import Library

import cantera

if int(cantera.__version__.replace('.', '')) >= 250:
    class Test(unittest.TestCase):
        def test(self):
            output_library = run()

            gold_file = abspath(join('tests',
                                     'tabulation',
                                     'adiabatic_slfm',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], rtol=1.e-6, atol=1.e-6))


    if __name__ == '__main__':
        unittest.main()
