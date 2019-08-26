import unittest
from numpy import exp, log, mean
from spitfire.time.methods import *
from spitfire.time.governor import Governor, FinalTime
from spitfire.time.nonlinear import SimpleNewtonSolver  # this is necessary despite what your IDE says


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

    def setup(self, state, prefactor):
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

    class SaveLastDataPoint(object):
        def __init__(self):
            self._last_t_value = None
            self._last_u_value = None

        @property
        def last_t_value(self):
            return self._last_t_value

        @property
        def last_u_value(self):
            return self._last_u_value

        def get(self, t, u, *args, **kwargs):
            self._last_t_value = t
            self._last_u_value = u

    data = SaveLastDataPoint()

    governor = Governor()
    governor.do_logging = False
    governor.termination_criteria = FinalTime(1.0)
    governor.custom_post_process_step = data.get

    dtlist = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    for dt in dtlist:
        governor.integrate(right_hand_side=rhs,
                           projector_setup=setup,
                           projector_solve=solve,
                           initial_condition=array([1.]),
                           method=method,
                           controller=dt)
        errors.append(norm(exp(-data.last_t_value) - data.last_u_value))

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


explicit_methods = ['ForwardEuler',
                    'ExplicitRungeKutta2Midpoint',
                    'ExplicitRungeKutta2Trapezoid',
                    'ExplicitRungeKutta2Ralston',
                    'ExplicitRungeKutta3Kutta',
                    'ExplicitRungeKutta4Classical',
                    'AdaptiveERK21HeunEuler',
                    'AdaptiveERK54CashKarp']

implicit_methods = ['BackwardEuler',
                    'BackwardEulerWithError',
                    'CrankNicolson',
                    'ImplicitMidpoint',
                    'SDIRK22',
                    'ESDIRK64']
nonlinear_solvers = ['SimpleNewtonSolver']


class TestOrderOfAccuracy(unittest.TestCase):
    pass


for method_name in explicit_methods:
    constructor = globals()[method_name]
    setattr(TestOrderOfAccuracy, 'test_' + method_name, create_test(constructor()))

for method_name in implicit_methods:
    method_constructor = globals()[method_name]
    for solver_name in nonlinear_solvers:
        solver_constructor = globals()[solver_name]
        setattr(TestOrderOfAccuracy,
                'test_' + method_name + '_' + solver_name,
                create_test(method_constructor(solver_constructor())))

if __name__ == '__main__':
    unittest.main()
