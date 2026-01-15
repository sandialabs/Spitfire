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

    def test_dissipation_rate(self):
        from spitfire import compute_dissipation_rate
        z = 0.5
        z_array = np.array([z])
        chi_max = 1.
        chi_st = 1.e-1
        z_st = 0.01

        try:
            chi = compute_dissipation_rate(z, chi_max)
            chi = compute_dissipation_rate(z_array, chi_max)
            self.assertTrue(True)
        except:
            self.assertTrue(False)

        for form in ['Peters', 'constant']:
            try:
                chi = compute_dissipation_rate(z, chi_max, form=form)
                chi = compute_dissipation_rate(z, form=form, stoich_dissipation_rate=chi_st, stoich_mixture_fraction=z_st)
                chi = compute_dissipation_rate(z_array, chi_max, form=form)
                chi = compute_dissipation_rate(z_array, form=form, stoich_dissipation_rate=chi_st, stoich_mixture_fraction=z_st)
                self.assertTrue(True)
            except:
                self.assertTrue(False)

        try:
            chi = compute_dissipation_rate(z)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        try:
            chi = compute_dissipation_rate(z,stoich_dissipation_rate=chi_st)
            self.assertTrue(False)
        except:
            self.assertTrue(True)

        try:
            chi = compute_dissipation_rate(z,stoich_dissipation_rate=chi_st, stoich_mixture_fraction=z_st)
            self.assertTrue(True)
        except:
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
