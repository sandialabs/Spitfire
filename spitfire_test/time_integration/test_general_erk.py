import unittest
from numpy import exp, log, mean, array, abs
from spitfire import GeneralAdaptiveERK, odesolve


class ExponentialDecayProblem(object):
    def __init__(self):
        self.decay_constant = -1.
        self.lhs_inverse = None

    def rhs(self, t, state):
        return self.decay_constant * state


def validate_method(method):
    edp = ExponentialDecayProblem()
    rhs = edp.rhs

    dtlist = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    tf = 1.0
    for dt in dtlist:
        qf = odesolve(rhs, array([1.]), array([tf]), method=method, step_size=dt)
        errors.append(abs(exp(-tf) - qf))

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


explicit_methods = [{'name': 'Forward Euler',
                     'A': array([[0.]]),
                     'b': array([1.]),
                     'order': 1},
                    {'name': 'Trapezoidal',
                     'A': array([[0., 0.],
                                 [1., 0.]]),
                     'b': array([0.5, 0.5]),
                     'order': 2},
                    {'name': 'Trapezoidal_with_Euler',
                     'A': array([[0., 0.],
                                 [1., 0.]]),
                     'b': array([0.5, 0.5]),
                     'bhat': array([1., 0.]),
                     'order': 2},
                    {'name': 'Midpoint',
                     'A': array([[0., 0.],
                                 [0.5, 0.]]),
                     'b': array([0., 1.]),
                     'order': 2},
                    {'name': 'Ralston',
                     'A': array([[0., 0.],
                                 [2. / 3., 0.]]),
                     'b': array([0.25, 0.75]),
                     'order': 2},
                    {'name': 'Classical RK4',
                     'A': array([[0., 0., 0., 0.],
                                 [0.5, 0., 0., 0.],
                                 [0., 0.5, 0., 0.],
                                 [0., 0., 1., 0.]]),
                     'b': array([1. / 6., 1. / 3., 1. / 3., 1. / 6.]),
                     'order': 4}]

for method_dict in explicit_methods:
    method = GeneralAdaptiveERK(name=method_dict['name'],
                                order=method_dict['order'],
                                A=method_dict['A'],
                                b=method_dict['b'],
                                bhat=None if 'bhat' not in method_dict else method_dict['bhat'])
    setattr(TestOrderOfAccuracy, 'test_general_erk_' + method_dict['name'], create_test(method))

if __name__ == '__main__':
    unittest.main()
