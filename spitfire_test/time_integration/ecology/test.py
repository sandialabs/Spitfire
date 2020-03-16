import pickle
import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from spitfire_test.time_integration.ecology.rebless import run


class Test(unittest.TestCase):
    def test(self):
        output = run()

        gold_file = abspath(join('spitfire_test',
                                 'time_integration',
                                 'ecology',
                                 'gold.pkl'))
        with open(gold_file, 'rb') as gold_input:
            gold_output = pickle.load(gold_input)

            for key in gold_output:
                t, sol = output[key]
                gold_t, gold_sol = gold_output[key]
                self.assertIsNone(assert_allclose(t, gold_t, atol=1.e-8))
                self.assertIsNone(assert_allclose(sol, gold_sol, atol=1.e-8))


if __name__ == '__main__':
    unittest.main()
