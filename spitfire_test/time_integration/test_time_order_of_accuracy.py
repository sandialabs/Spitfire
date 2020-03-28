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
    if isinstance(method, ImplicitTimeStepper):
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
                      array([1.]),
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

    return success


def create_test(m):
    def test(self):
        self.assertTrue(validate_method(m))

    return test


class TestOrderOfAccuracy(unittest.TestCase):
    pass


for method in [ForwardEuler,
               ExplicitRungeKutta2Midpoint,
               ExplicitRungeKutta2Trapezoid,
               ExplicitRungeKutta2Ralston,
               ExplicitRungeKutta3Kutta,
               ExplicitRungeKutta4Classical,
               AdaptiveERK21HeunEuler,
               AdaptiveERK54CashKarp]:
    setattr(TestOrderOfAccuracy, 'test_' + str(method), create_test(method()))

for method in [BackwardEuler,
               BackwardEulerWithError,
               CrankNicolson,
               SDIRK22,
               SDIRK22KForm,
               ESDIRK64]:
    for solver in [SimpleNewtonSolver]:
        setattr(TestOrderOfAccuracy, 'test_' + str(method) + '_' + str(solver), create_test(method(solver())))

if __name__ == '__main__':
    unittest.main()
