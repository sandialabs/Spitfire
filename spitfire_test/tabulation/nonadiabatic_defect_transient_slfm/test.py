import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from spitfire_test.tabulation.nonadiabatic_defect_transient_slfm.rebless import run
from spitfire.chemistry.library import Library

import cantera

if int(cantera.__version__.replace('.', '')) >= 250:
    class Test(unittest.TestCase):
        def test_serial(self):
            output_library = run(num_procs=1)

            gold_file = abspath(join('spitfire_test',
                                     'tabulation',
                                     'nonadiabatic_defect_transient_slfm',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], rtol=2.e-4, atol=1.e-4))

        def test_parallel(self):
            output_library = run(num_procs=2)

            gold_file = abspath(join('spitfire_test',
                                     'tabulation',
                                     'nonadiabatic_defect_transient_slfm',
                                     'gold.pkl'))
            gold_library = Library.load_from_file(gold_file)

            for prop in gold_library.props:
                self.assertIsNone(assert_allclose(gold_library[prop], output_library[prop], rtol=2.e-4, atol=1.e-4))


    if __name__ == '__main__':
        unittest.main()
