import unittest
from spitfire import SimpleNewtonSolver
from numpy import abs, imag, any, Inf, zeros, array, NaN
from numpy import copy as numpy_copy


def direct_residual(fun):
    def doubled(x, *args, **kwargs):
        output = fun(x, *args, **kwargs)
        return output, numpy_copy(output)

    return doubled


def direct_solve(fun):
    def append_iteration_count_of_one_and_converged(x, *args, **kwargs):
        output = fun(x, *args, **kwargs)
        return output, 1, True

    return append_iteration_count_of_one_and_converged


def make_sure_is_real(solution, *args, **kwargs):
    return True if any(imag(solution)) else False


@direct_residual
def linear_problem_residual(x, *args, **kwargs):
    return 2. * x - 1.


def linear_problem_jacobian_inverse(x, *args, **kwargs):
    return 0.5


@direct_solve
def linear_problem_solve_free(resid, *args, **kwargs):
    return 0.5 * resid


class LinearProblem(object):
    def __init__(self):
        self.lhs_inverse = None

    def residual(self, x, *args, **kwargs):
        return linear_problem_residual(x, args, kwargs)

    def setup(self, x, *args, **kwargs):
        self.lhs_inverse = linear_problem_jacobian_inverse(x, args, kwargs)

    @direct_solve
    def solve(self, resid, *args, **kwargs):
        return self.lhs_inverse * resid


class LinearScalarTest(unittest.TestCase):
    def test_frozen_jacobian(self):
        guess = 0.3

        solution = 0.5
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(tolerance=tolerance,
                                    max_nonlinear_iter=4,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=2,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        newton.slowness_detection_iter = 2  # to spitfire_test detection of slow convergence

        output = newton(residual_method=linear_problem_residual,
                        solve_method=linear_problem_solve_free,
                        initial_guess=guess,
                        initial_rhs=linear_problem_residual(guess),
                        setup_method=None)

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((linear_problem_residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 1)
        self.assertTrue(output.liter == 1)
        self.assertTrue(output.projector_setups == 0)
        self.assertTrue(not output.slow_convergence)
        self.assertTrue(output.converged)

    def test_active_jacobian(self):
        problem = LinearProblem()
        guess = 0.3

        solution = 0.5
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=True,
                                    tolerance=tolerance,
                                    max_nonlinear_iter=4,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        newton.slowness_detection_iter = 0  # to spitfire_test detection of slow convergence

        output = newton(residual_method=problem.residual,
                        setup_method=problem.setup,
                        solve_method=problem.solve,
                        initial_guess=guess,
                        initial_rhs=problem.residual(guess))

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((problem.residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 1)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.projector_setups == output.iter)
        self.assertTrue(output.converged)
        self.assertTrue(output.slow_convergence)


@direct_residual
def quadratic_problem_residual(x, *args, **kwargs):
    return x * x - 4.


def quadratic_problem_jacobian_inverse(x, *args, **kwargs):
    return 0.5 / x


def quadratic_problem_solve_free(resid, x, *args, **kwargs):
    return quadratic_problem_jacobian_inverse(x) * resid


class QuadraticProblem(object):
    def __init__(self):
        self.lhs_inverse = None

    def residual(self, x, *args, **kwargs):
        return quadratic_problem_residual(x, args, kwargs)

    def setup(self, x, *args, **kwargs):
        self.lhs_inverse = quadratic_problem_jacobian_inverse(x, args, kwargs)

    @direct_solve
    def solve(self, resid, *args, **kwargs):
        return self.lhs_inverse * resid


class QuadraticScalarTest(unittest.TestCase):
    def test_frozen_jacobian(self):
        guess = 1.6

        @direct_solve
        def solve_with_frozen_jacobian(resid, *args, **kwargs):
            return quadratic_problem_solve_free(resid, guess)

        solution = 2.
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(tolerance=tolerance,
                                    max_nonlinear_iter=25,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        output = newton(residual_method=quadratic_problem_residual,
                        solve_method=solve_with_frozen_jacobian,
                        initial_guess=guess,
                        initial_rhs=quadratic_problem_residual(guess),
                        setup_method=None)

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((quadratic_problem_residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 20)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.converged)

    def test_active_jacobian(self):
        problem = QuadraticProblem()

        guess = 1.6

        solution = 2.
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=True,
                                    tolerance=tolerance,
                                    max_nonlinear_iter=10,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        output = newton(residual_method=problem.residual,
                        solve_method=problem.solve,
                        setup_method=problem.setup,
                        initial_guess=guess,
                        initial_rhs=problem.residual(guess))

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((problem.residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 4)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.projector_setups == output.iter)
        self.assertTrue(output.converged)


@direct_residual
def cubic_problem_residual(x, *args, **kwargs):
    return x * x * x - 27.


def cubic_problem_jacobian_inverse(x, *args, **kwargs):
    return 1. / (3. * x * x)


def cubic_problem_solve_free(resid, x, *args, **kwargs):
    return cubic_problem_jacobian_inverse(x) * resid


class CubicProblem(object):
    def __init__(self):
        self.lhs_inverse = None

    def residual(self, x, *args, **kwargs):
        return cubic_problem_residual(x, args, kwargs)

    def setup(self, x, *args, **kwargs):
        self.lhs_inverse = cubic_problem_jacobian_inverse(x, args, kwargs)

    @direct_solve
    def solve(self, resid, *args, **kwargs):
        return self.lhs_inverse * resid


class CubicScalarTest(unittest.TestCase):
    def test_frozen_jacobian(self):
        guess = 2.6

        @direct_solve
        def solve_with_frozen_jacobian(resid, *args, **kwargs):
            return cubic_problem_solve_free(resid, guess)

        solution = 3.
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(tolerance=tolerance,
                                    max_nonlinear_iter=40,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        output = newton(residual_method=cubic_problem_residual,
                        solve_method=solve_with_frozen_jacobian,
                        initial_guess=guess,
                        initial_rhs=cubic_problem_residual(guess),
                        setup_method=None)

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((cubic_problem_residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 27)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.converged)

    def test_active_jacobian(self):
        problem = CubicProblem()

        guess = 2.6

        solution = 3.
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=True,
                                    tolerance=tolerance,
                                    max_nonlinear_iter=10,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        output = newton(residual_method=problem.residual,
                        solve_method=problem.solve,
                        setup_method=problem.setup,
                        initial_guess=guess,
                        initial_rhs=problem.residual(guess))

        self.assertTrue(abs(output.solution - solution) <= tolerance)
        self.assertTrue((problem.residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 5)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.projector_setups == output.iter)
        self.assertTrue(output.converged)

    def test_failure_catching(self):
        guess = 2.6

        @direct_solve
        def solve_with_frozen_jacobian(resid, *args, **kwargs):
            return cubic_problem_solve_free(resid, guess)

        solution = 3.
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(tolerance=tolerance,
                                    max_nonlinear_iter=40,
                                    must_converge=True,
                                    norm_weighting=1.,
                                    norm_order=Inf,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)

        # spitfire_test that if given 0 iterations to converge, it cannot
        newton.max_nonlinear_iter = 0
        try:
            newton(residual_method=cubic_problem_residual,
                   solve_method=solve_with_frozen_jacobian,
                   initial_guess=guess,
                   initial_rhs=cubic_problem_residual(guess),
                   setup_method=None)
            self.assertTrue(False, 'Newton incorrectly did not recognize 0 iteration limit')
        except ValueError:
            self.assertTrue(True)

        # spitfire_test that if we set must_converge=False, that it doesn't raise an error for not converging in 0 iterations,
        # and that it marks the failure to converge correctly in the output
        newton.must_converge = False
        try:
            output = newton(residual_method=cubic_problem_residual,
                            solve_method=solve_with_frozen_jacobian,
                            initial_guess=guess,
                            initial_rhs=cubic_problem_residual(guess),
                            setup_method=None)
            self.assertTrue(not output.converged)
        except ValueError:
            self.assertTrue(False, 'Newton incorrectly said it failed to converge but we said must_converge=False')

        newton.max_nonlinear_iter = 40  # reset
        newton.must_converge = True  # reset

        # spitfire_test that if we give it a NaN with raise_naninf=True (set above), it does in fact raise an error
        guess = NaN
        try:
            newton(residual_method=cubic_problem_residual,
                   solve_method=solve_with_frozen_jacobian,
                   initial_guess=guess,
                   initial_rhs=cubic_problem_residual(guess),
                   setup_method=None)
            self.assertTrue(False, 'Newton ate a NaN and did not catch it despite raise_naninf=True')
        except ValueError:
            self.assertTrue(True)

        # spitfire_test that if we give it an Inf with raise_naninf=True (set above), it does in fact raise an error
        guess = Inf
        try:
            newton(residual_method=cubic_problem_residual,
                   solve_method=solve_with_frozen_jacobian,
                   initial_guess=guess,
                   initial_rhs=cubic_problem_residual(guess),
                   setup_method=None)
            self.assertTrue(False, 'Newton ate an Inf and did not catch it despite raise_naninf=True')
        except ValueError:
            self.assertTrue(True)


class VectorProblem(object):
    def __init__(self):
        self.lhs_inverse = zeros(3)
        self.norm_weighting = array([1., 2., 3.])

    @direct_residual
    def residual(self, x, *args, **kwargs):
        return array([linear_problem_residual(x[0], args, kwargs)[0],
                      quadratic_problem_residual(x[1], args, kwargs)[0],
                      cubic_problem_residual(x[2], args, kwargs)[0]])

    def setup(self, x, *args, **kwargs):
        self.lhs_inverse = array([linear_problem_jacobian_inverse(x[0], args, kwargs),
                                  quadratic_problem_jacobian_inverse(x[1], args, kwargs),
                                  cubic_problem_jacobian_inverse(x[2], args, kwargs)])

    @direct_solve
    def solve(self, resid, *args, **kwargs):
        return self.lhs_inverse * resid


class VectorTest(unittest.TestCase):
    def test_active_jacobian(self):
        problem = VectorProblem()

        guess = array([0.6, 1.6, 2.6])

        solution = array([0.5, 2., 3.])
        tolerance = 1.e-12

        newton = SimpleNewtonSolver(evaluate_jacobian_every_iter=True,
                                    tolerance=tolerance,
                                    max_nonlinear_iter=10,
                                    must_converge=True,
                                    norm_weighting=problem.norm_weighting,
                                    norm_order=2,
                                    raise_naninf=True,
                                    custom_solution_check=make_sure_is_real)
        output = newton(residual_method=problem.residual,
                        solve_method=problem.solve,
                        setup_method=problem.setup,
                        initial_guess=guess,
                        initial_rhs=problem.residual(guess))

        self.assertTrue((abs(output.solution - solution) <= tolerance).all())
        self.assertTrue((problem.residual(output.solution) == output.rhs_at_converged).all())
        self.assertTrue(output.iter == 5)
        self.assertTrue(output.liter == output.iter)
        self.assertTrue(output.projector_setups == output.iter)
        self.assertTrue(output.converged)


# todo: add tests that cover failure-catching


if __name__ == '__main__':
    unittest.main()
