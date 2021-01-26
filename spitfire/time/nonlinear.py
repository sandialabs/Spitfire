"""
This module contains nonlinear solvers used in time stepping.
At the moment this is simply Newton's method.
"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

import numpy as np
from numpy import copy as numpy_copy
from numpy import any, logical_or, isinf, isnan, abs, Inf
from scipy.linalg import norm


def finite_difference_jacobian(residual_func, residual_value, state, offset_rel=1.e-5, offset_abs=1.e-8):
    """
    Compute a simple one-sided finite difference approximation to the Jacobian of a residual function.

    Parameters
    ----------
    residual_func : callable f(q)->r
        residual function of the state vector, r(q)
    residual_value : np.array
        value of the residual function at the specified state vector
    state : np.array
        the state vector
    offset_rel : float
        the relative contribution to the finite difference delta (optional, default: 1e-5)
    offset_abs : float
        the absolute contribution to the finite difference delta (optional, default: 1e-8)
    Returns
    -------
    j : np.ndarray
        the approximate Jacobian matrix

    """
    neq = state.size
    j = np.ndarray((neq, neq))
    for i in range(neq):
        state_offset = numpy_copy(state)
        offset = offset_rel * np.abs(state[i]) + offset_abs
        state_offset[i] += offset
        j[:, i] = (residual_func(state_offset) - residual_value) / offset
    return j


class SolverOutput(object):
    """Read-only class that holds information about the result of a nonlinear solver.

    **Constructor**: build a SolverOutput object, specifying all data here (the object will be read-only)

    Parameters
    ----------
    solution : np.ndarray
        the solution to the nonlinear system
    iter : int
        how many nonlinear iterations were needed for convergence
    liter : int
        how many total linear iterations were needed for convergence
    converged : bool
        whether or not the solver converged to a solution
    slow_convergence : bool
        whether or not the solver detected slow convergence
    projector_setups : int
        the number of times the linear projector was set up (e.g. Jacobian evaluation-factorization)
    rhs_at_converged : np.ndarray
        the right-hand side of the ODE system at the converged solution

    """

    __slots__ = ['solution',
                 'rhs_at_converged',
                 'iter',
                 'liter',
                 'converged',
                 'slow_convergence',
                 'projector_setups']

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            self.__setattr__(slot, kwargs[slot] if slot in kwargs else None)


class NonlinearSolver(object):
    """Base class for nonlinear solvers.

    **Constructor**: build a NonlinearSolver object

    Parameters
    ----------
    max_nonlinear_iter : int
        maximum number of nonlinear iterations to try (default: 20)
    slowness_detection_iter : int
        how many iterations the solver runs can run without declaring convergence "slow" (default: Inf)
    must_converge : bool
        whether or not the solver must converge to a solution within max_nonlinear_iter (default: False)
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    raise_naninf : bool
        whether or not to check for NaN/Inf values in the solution and residual and raise an exception if found (default: False)
    custom_solution_check : callable
        a function of the solution that executes custom checks for solution validity (default: None)
    setup_projector_in_governor : bool
        whether or not the linear projector is set up outside of the solver (default: True)

    """

    defaults = {'max_nonlinear_iter': 20,
                'slowness_detection_iter': Inf,
                'must_converge': False,
                'tolerance': 1.e-12,
                'norm_weighting': 1.,
                'norm_order': Inf,
                'raise_naninf': False,
                'custom_solution_check': None,
                'setup_projector_in_governor': True}

    def __init__(self, *args, **kwargs):
        for attr in self.defaults:
            self.__setattr__(attr, kwargs[attr] if attr in kwargs else self.defaults[attr])

    @staticmethod
    def _there_are_any_naninf(state):
        return any(logical_or(isinf(state), isnan(state)))

    def _check_for_naninf(self, variable, message):
        if self.raise_naninf:
            if self._there_are_any_naninf(variable):
                raise ValueError(message)

    def _run_custom_solution_check(self, solution, debug_message):
        if self.custom_solution_check is not None:
            self.custom_solution_check(solution, debug_message)


class SimpleNewtonSolver(NonlinearSolver):
    """Simple Newton solver

    **Constructor**: build a SimpleNewtonSolver object

    Parameters
    ----------
    evaluate_jacobian_every_iter : bool
        whether or not to set up the linear projector on every iteration of Newton's method (default: False)
    max_nonlinear_iter : int
        maximum number of nonlinear iterations to try (default: 20)
    slowness_detection_iter : int
        how many iterations the solver runs can run without declaring convergence "slow" (default: Inf)
    must_converge : bool
        whether or not the solver must converge to a solution within max_nonlinear_iter (default: False)
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    raise_naninf : bool
        whether or not to check for NaN/Inf values in the solution and residual and raise an exception if found (default: False)
    custom_solution_check : callable
        a function of the solution that executes custom checks for solution validity (default: None)
    setup_projector_in_governor : bool
        whether or not the linear projector is set up outside of the solver (default: True)

    """

    def __init__(self, evaluate_jacobian_every_iter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evaluate_jacobian_every_iter = evaluate_jacobian_every_iter
        self.setup_projector_in_governor = not self._evaluate_jacobian_every_iter

    @property
    def evaluate_jacobian_every_iter(self):
        return self._evaluate_jacobian_every_iter

    @evaluate_jacobian_every_iter.setter
    def evaluate_jacobian_every_iter(self, newvalue):
        self._evaluate_jacobian_every_iter = newvalue
        self.setup_projector_in_governor = not self._evaluate_jacobian_every_iter

    def __call__(self,
                 residual_method,
                 setup_method,
                 solve_method,
                 initial_guess,
                 initial_rhs):
        """
        Solve a nonlinear problem and return the result.

        Parameters
        ----------
        residual_method : callable f(x)->(res,rhs)
            residual function of the solution that returns both the residual and right-hand side of an ODE,
            to use this on a non-ODE problem simply have your residual method return as residual, None
        setup_method : callable f(x)
            setup function of the solution for the linear projector (e.g. Jacobian eval and factorize)
        solve_method : callabe f(x)->b
            linear projector solver function of a residual alone,
            which returns the state update, linear solver iterations, and whether or not the linear solver converged
        initial_guess : np.array
            initial guess for the solution
        initial_rhs : np.array
            ODE right-hand side for the problem at the initial guess (to avoid re-evaluation)
        Returns
        -------
        sol : np.array
            solution to the nonlinear problem

        """

        solution = numpy_copy(initial_guess)
        self._check_for_naninf(solution, 'NaN or Inf detected in Simple Newton Solve: in the initial solution!')
        self._run_custom_solution_check(solution, 'In the initial solution')

        this_is_a_scalar_problem = True if solution.size == 1 else False
        norm_method = abs if this_is_a_scalar_problem else lambda x: norm(x, ord=self.norm_order)

        residual, rhs = residual_method(solution, existing_rhs=initial_rhs, evaluate_new_rhs=False)
        self._check_for_naninf(residual, 'NaN or Inf detected in Simple Newton Solve: in the initial residual!')

        projector_setups = 0
        total_linear_iter = 0

        for iteration_count in range(1, self.max_nonlinear_iter + 1):
            if self.evaluate_jacobian_every_iter:
                setup_method(solution)
                projector_setups += 1
            dstate, this_linear_solver_iter, linear_converged = solve_method(residual)

            debug_string = 'On iteration ' + str(iteration_count) + ', linear solve convergence: ' + str(
                linear_converged) + ' in ' + str(this_linear_solver_iter) + ' iterations'

            self._check_for_naninf(dstate,
                                   'NaN or Inf detected in Simple Newton Solve: solution update check! ' + debug_string)
            total_linear_iter += this_linear_solver_iter
            solution -= dstate
            self._check_for_naninf(solution,
                                   'NaN or Inf detected in Simple Newton Solve: solution check! ' + debug_string)
            self._run_custom_solution_check(solution, debug_string)

            residual, rhs = residual_method(solution, evaluate_new_rhs=True)
            self._check_for_naninf(residual,
                                   'NaN or Inf detected in Simple Newton Solve: solution check! ' + debug_string)

            if norm_method(residual * self.norm_weighting) < self.tolerance:
                return SolverOutput(slow_convergence=iteration_count > self.slowness_detection_iter,
                                    solution=solution,
                                    rhs_at_converged=rhs,
                                    iter=iteration_count,
                                    liter=total_linear_iter,
                                    converged=True,
                                    projector_setups=projector_setups)

        if self.must_converge:
            raise ValueError('Simple Newton method did not converge and must_converge=True!')
        else:
            return SolverOutput(
                solution=solution,
                rhs_at_converged=rhs,
                iter=self.max_nonlinear_iter,
                liter=total_linear_iter,
                converged=False,
                slow_convergence=True,
                projector_setups=projector_setups)
