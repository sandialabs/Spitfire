import pickle
import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from spitfire_test.time_integration.chemistry_abc_explicit.rebless import run


class Test(unittest.TestCase):
    def test(self):
        output = run()

        gold_file = abspath(join('spitfire_test',
                                 'time_integration',
                                 'chemistry_abc_explicit',
                                 'gold.pkl'))
        with open(gold_file, 'rb') as gold_input:
            gold_output = pickle.load(gold_input)
            self.assertIsNone(assert_allclose(output['t'], gold_output['t'], atol=1.e-8))
            self.assertIsNone(assert_allclose(output['sol'], gold_output['sol'], atol=1.e-8))


if __name__ == '__main__':
    unittest.main()
