import unittest
from spitfire import Library, Dimension
import numpy as np


class TestDimensionExceptions(unittest.TestCase):
    def test_check_for_invalid_python_names_hyphen(self):
        try:
            x = Dimension('x-1', np.linspace(0, 1, 16))
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_check_for_invalid_python_space(self):
        try:
            x = Dimension('x 1', np.linspace(0, 1, 16))
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_check_for_nonflat_data(self):
        try:
            x = Dimension('x', np.zeros((4, 2)))
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_check_for_duplicate_data(self):
        try:
            x = Dimension('x', np.array([0, 1, 1, 2]))
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
