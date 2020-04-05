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


class ExponentialDecayProblem(object):
    def __init__(self):
        self.decay_constant = -1.
        self.lhs_inverse = None

    def rhs(self, t, state):
        return self.decay_constant * state

    def setup(self, t, state, prefactor):
        self.lhs_inverse = 1. / (prefactor * self.decay_constant - 1.)

    @direct_solve
    def solve(self, residual):
        return self.lhs_inverse * residual


def validate_method(method):
    edp = ExponentialDecayProblem()
    rhs = edp.rhs
    if method.is_implicit:
        setup = edp.setup
        solve = edp.solve
    else:
        setup = None
        solve = None

    dtlist = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    tf = 1.0
    for dt in dtlist:
        qf = odesolve(rhs,
                      array([1.]),
                      array([tf]),
                      linear_setup=setup,
                      linear_solve=solve,
                      method=method,
                      step_size=dt)
        errors.append(norm(exp(-tf) - qf))

    order_list = []
    for idx in range(len(errors) - 1):
        order_list.append(log(errors[idx] / errors[idx + 1]) / log(dtlist[idx] / dtlist[idx + 1]))

    observed_order = mean(array(order_list))
    success = method.order - 0.1 < observed_order < method.order + 0.1

    if not success:
        print(method.name, method.order, observed_order)

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
               BogackiShampineS4P3Q2,
               ZonneveldS5P4Q3,
               ExpKennedyCarpetnerS6P4Q3,
               CashKarpS6P5Q4, ]:
    setattr(TestOrderOfAccuracy, 'test_' + str(method), create_test(method()))

for method in [BackwardEulerS1P1Q1,
               CrankNicolsonS2P2,
               KennedyCarpenterS6P4Q3,
               KennedyCarpenterS4P3Q2,
               KvaernoS4P3Q2,
               KennedyCarpenterS8P5Q4, ]:
    for solver in [SimpleNewtonSolver]:
        setattr(TestOrderOfAccuracy, 'test_' + str(method) + '_' + str(solver), create_test(method(solver())))

if __name__ == '__main__':
    unittest.main()
