import unittest
import numpy as np
from spitfire.time.integrator import odesolve, FailedODESolveException


class TestMaximumResidual(unittest.TestCase):
    def test_maxres(self):
        maxres = 1.
        tcutoff = 1.
        X0 = np.array([1.])

        def rhs(t, x):
            return -0.1*x if t<tcutoff else np.array([maxres*1.1])

        try:
            ts, xs = odesolve(rhs, X0, stop_at_time=tcutoff, save_each_step=True, maximum_residual=maxres, print_exception_on_failure=False)
            self.assertTrue(True)
        except FailedODESolveException as exc:
            self.assertTrue(False)

        try:
            ts, xs = odesolve(rhs, X0, stop_at_time=tcutoff+0.1, save_each_step=True, maximum_residual=maxres, print_exception_on_failure=False)
            self.assertTrue(False)
        except FailedODESolveException as exc:
            ts, xs = exc.times, exc.states
            self.assertTrue(True)
            self.assertTrue(ts[-1]>=tcutoff)
            self.assertTrue(ts[-2]<tcutoff)


if __name__ == '__main__':
    unittest.main()
