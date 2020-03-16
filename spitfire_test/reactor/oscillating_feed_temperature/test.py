import pickle
import unittest
from os.path import join, abspath

from numpy.testing import assert_allclose

from spitfire_test.reactor.oscillating_feed_temperature.rebless import run


class Test(unittest.TestCase):
    def test(self):
        output = run()

        gold_file = abspath(join('spitfire_test',
                                 'reactor',
                                 'oscillating_feed_temperature',
                                 'gold.pkl'))
        with open(gold_file, 'rb') as gold_input:
            gold_output = pickle.load(gold_input)

            t, T, Y = output
            gold_t, gold_T, gold_Y = gold_output
            self.assertIsNone(assert_allclose(t, gold_t, atol=1.e-8))
            self.assertIsNone(assert_allclose(T, gold_T, atol=1.e-8))
            self.assertIsNone(assert_allclose(Y, gold_Y, atol=1.e-8))


if __name__ == '__main__':
    unittest.main()
