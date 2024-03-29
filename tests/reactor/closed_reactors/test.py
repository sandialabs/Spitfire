import pickle
import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from tests.reactor.closed_reactors.rebless import run
from spitfire.chemistry.ctversion import check as cantera_version_check


if cantera_version_check('atleast', 2, 5, None):
    class Test(unittest.TestCase):
        def test(self):
            output = run()

            gold_file = abspath(join('tests',
                                     'reactor',
                                     'closed_reactors',
                                     'gold.pkl'))
            with open(gold_file, 'rb') as gold_input:
                gold_output = pickle.load(gold_input)

                for key in output:
                    t, T, Y = output[key]
                    gold_t, gold_T, gold_Y = gold_output[key]
                    self.assertIsNone(assert_allclose(t, gold_t, atol=1.e-8, rtol=1.e-4))
                    self.assertIsNone(assert_allclose(T, gold_T, atol=1.e-8, rtol=1.e-4))
                    self.assertIsNone(assert_allclose(Y, gold_Y, atol=1.e-8, rtol=1.e-4))


    if __name__ == '__main__':
        unittest.main()
