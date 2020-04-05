import unittest
from numpy import exp, log, mean
from spitfire.time.methods import *
from spitfire.time.nonlinear import *
from spitfire import odesolve


def direct_solve(fun):
    def append_iteration_count_of_one_and_converged(x, *args, **kwargs):
        output = fun(x, *args, **kwargs)
        return output, 1, True

    return append_iteration_count_of_one_and_converged


def validate_method(method):
    theta = 3.03623184819656
    q0 = np.array([0., theta * np.tanh(0.25 * theta)])
    rhs = lambda t, q: np.array([q[1], -np.exp(1. + q[0])])

    dtlist = [0.06, 0.02, 0.01, 0.006, 0.004]
    errors = []
    tf = 1.0
    for dt in dtlist:
        qf = odesolve(rhs,
                      q0,
                      array([tf]),
                      method=method,
                      step_size=dt)
        errors.append(norm(qf[0, 0]))

    order_list = []
    for idx in range(len(errors) - 1):
        order_list.append(log(errors[idx] / errors[idx + 1]) / log(dtlist[idx] / dtlist[idx + 1]))

    observed_order = mean(array(order_list))
    success = method.order - 0.1 < observed_order < method.order + 0.1

    return success


def create_test(m):
    def test(self):
        self.assertTrue(validate_method(m))

    return test


class TestOrderOfAccuracy(unittest.TestCase):
    pass


for method in [ForwardEulerS1P1,
               ExpMidpointS2P2,
               ExpTrapezoidalS2P2Q1,
               ExpRalstonS2P2,
               RK3KuttaS3P3,
               RK4ClassicalS4P4,
               CashKarpS6P5Q4, ]:
    setattr(TestOrderOfAccuracy, 'test_' + str(method), create_test(method()))

for method in [BackwardEulerS1P1Q1,
               CrankNicolsonS2P2,
               SDIRKS2P2,
               KennedyCarpenterS6P4Q3]:
    for solver in [SimpleNewtonSolver]:
        setattr(TestOrderOfAccuracy, 'test_' + str(method) + '_' + str(solver), create_test(method(solver())))

if __name__ == '__main__':
    unittest.main()
