import unittest
import numpy as np
from scipy import integrate as scipy_integrate
from spitfire.time.integrator import odesolve


class TestComparisonToSciPyIntegrate(unittest.TestCase):
    def test_lotka_volterra(self):
        a = 1.
        b = 0.1
        c = 1.5
        d = 0.75

        def rhs(t, x):
            return np.array([a * x[0] - b * x[0] * x[1],
                             -c * x[1] + d * b * x[0] * x[1]])

        t = np.linspace(0., 10., 40)
        X0 = np.array([10., 5.])

        X, infodict = scipy_integrate.odeint(lambda xv, tv: rhs(tv, xv), X0, t, full_output=True)
        Xs, stats = odesolve(rhs, X0, t, return_info=True)

        self.assertTrue(np.max(np.abs((X - Xs) / X)) < 1e-6)


if __name__ == '__main__':
    unittest.main()
