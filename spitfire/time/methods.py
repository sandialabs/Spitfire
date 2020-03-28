"""
This module contains time stepping methods that represent distinct methods of taking a single step in the time integration loop.
Some are explicit, some are implicit.
Some allow adaptive time stepping via control of embedded temporal error estimates while most do not.

Consider a general purpose explicit ODE,

.. math::
 \\frac{\\partial \\boldsymbol{q}}{\\partial t} = \\boldsymbol{r}(t, \\boldsymbol{q}),

where :math:`\\boldsymbol{q}=[q_1,q_2,\\ldots]` is the vector of state variables
and :math:`\\boldsymbol{r}=[r_1,r_2,\\ldots]` is the vector of right-hand side functions.
The classes in this module have a ``single_step(q, t, h, r, ...)`` method that steps from time level :math:`t^n` to
the next time,  :math:`t^{n+1}=t^n + h`, given a state :math:`q^n` and right-hand side function :math:`r(t,q)`.
For example, the forward Euler method with class ``ForwardEuler`` computes this as :math:`q^{n+1}=q^n + hr(t^n,q^n)`.

In Spitfire's unit testing we verify the order of accuracy of most of these methods.
If you add a new one be sure to add it to the unit tester.
See the file: ``spitfire_test/time/test_time_order_of_accuracy.py``.

"""

# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

from numpy import sqrt, copy, array, sum, Inf, zeros_like
from scipy.linalg import norm


class StepOutput(object):
    """Read-only class that holds information about the result of a single time step.

    **Constructor**: build a StepOutput object, specifying all data here (the object will be read-only)

    Parameters
    ----------
    solution_update : np.ndarray
        the update to the solution, the state at the next time level minus that at the current time level
    temporal_error : float
        the temporal error estimate - for adaptive time stepping through classical error control
    nonlinear_iter : int
        how many nonlinear iterations the step required - implicit methods only
    linear_iter : int
        how many linear iterations the step required - implicit methods only
    nonlinear_converged : bool
        whether or not the nonlinear solver converged - implicit methods only
    slow_nonlinear_convergence : bool
        whether or not the nonlinear solver detected slow convergence - implicit methods only
    projector_setups : int
        the number of times the linear projector was set up (e.g. Jacobian evaluation-factorization)
    extra_errors : list[float]
        a list of additional embedded temporal error estimates - for adaptive time stepping through new ratio control
    """
    __slots__ = ['solution_update',
                 'temporal_error',
                 'nonlinear_iter',
                 'linear_iter',
                 'nonlinear_converged',
                 'slow_nonlinear_convergence',
                 'projector_setups',
                 'extra_errors']

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            self.__setattr__(slot, kwargs[slot] if slot in kwargs else None)
        self.temporal_error = -1. if self.temporal_error is None else self.temporal_error


class TimeStepper(object):
    """
    Base class for time stepper classes

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper, e.g. 'Forward Euler' or 'Runge Kutta 4'
    order : int
        order of accuracy of the time stepping method
    """

    def __init__(self, name, order):
        self._name = name
        self._order = order

    @property
    def name(self):
        return self._name

    @property
    def order(self):
        return self._order


class ImplicitTimeStepper(TimeStepper):
    """
    Base class for implicit time stepper classes

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper
    order : int
        order of accuracy of the time stepping method
    implicit_coefficient : float
        the leading coefficient used in time-augmented Jacobian matrix assembly
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    """

    def __init__(self, name, order, implicit_coefficient, nonlinear_solver):
        super().__init__(name, order)
        self._implicit_coefficient = implicit_coefficient
        self._nonlinear_solver = nonlinear_solver

    @property
    def implicit_coefficient(self):
        return self._implicit_coefficient

    @property
    def nonlinear_solver(self):
        return self._nonlinear_solver


class AdaptiveImplicitTimeStepper(ImplicitTimeStepper):
    """
    Base class for adaptive implicit time stepper classes

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper
    order : int
        order of accuracy of the time stepping method
    implicit_coefficient : float
        the leading coefficient used in time-augmented Jacobian matrix assembly
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, name, order, implicit_coefficient, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name, order, implicit_coefficient, nonlinear_solver)
        self._norm_weighting = norm_weighting
        self._norm_order = norm_order

    @property
    def implicit_coefficient(self):
        return self._implicit_coefficient

    @property
    def nonlinear_solver(self):
        return self._nonlinear_solver

    @property
    def norm_weighting(self):
        return self._norm_weighting

    @property
    def norm_order(self):
        return self._norm_order


class ExplicitRungeKutta(TimeStepper):
    """
    Base class for explicit Runge-Kutta methods

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper, e.g. 'Forward Euler' or 'Runge Kutta 4'
    order : int
        order of accuracy of the time stepping method
    stages : int
        number of stages in the Runge-Kutta method
    """

    def __init__(self, name, order, stages):
        super().__init__(name, order)
        self._stages = stages

    @property
    def stages(self):
        return self._stages


class AdaptiveExplicitRungeKutta(ExplicitRungeKutta):
    """
    Base class for explicit Runge-Kutta methods

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper, e.g. 'Forward Euler' or 'Runge Kutta 4'
    order : int
        order of accuracy of the time stepping method
    error_order : int
        order of accuracy of the embedded error estimation
    stages : int
        number of stages in the Runge-Kutta method
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, name, order, error_order, stages, norm_weighting=1., norm_order=Inf):
        super().__init__(name, order, stages)
        self._error_order = error_order
        self._norm_weighting = norm_weighting
        self._norm_order = norm_order

    @property
    def error_order(self):
        return self._error_order

    @property
    def norm_weighting(self):
        return self._norm_weighting

    @property
    def norm_order(self):
        return self._norm_order


class ForwardEuler(TimeStepper):
    """
    The Forward Euler method
    """

    def __init__(self):
        super().__init__(name='Forward Euler', order=1)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        return StepOutput(solution_update=dt * rhs(t, state))


class ExplicitRungeKutta2Midpoint(ExplicitRungeKutta):
    """
    The explicit Midpoint method
    """

    def __init__(self):
        super().__init__(name='Explicit Midpoint RK2', order=2, stages=2)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        return StepOutput(solution_update=rhs(t + 0.5 * dt, state + 0.5 * dt * rhs(t, state)) * dt)


class ExplicitRungeKutta2Trapezoid(ExplicitRungeKutta):
    """
    The explicit Trapezoidal method
    """

    def __init__(self):
        super().__init__(name='Explicit Trapezoidal RK2', order=2, stages=2)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        k1 = rhs(t, state)
        return StepOutput(solution_update=0.5 * (k1 + rhs(t + dt, state + dt * k1)) * dt)


class ExplicitRungeKutta2Ralston(ExplicitRungeKutta):
    """
    The explicit Ralston method, the two-stage, second-order ERK method with lowest principal error norm
    """

    def __init__(self):
        super().__init__(name='Explicit Ralston RK2', order=2, stages=2)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        k1 = rhs(t, state)
        k2 = rhs(t + 2. / 3. * dt, state + 2. / 3. * dt * k1)
        return StepOutput(solution_update=0.25 * (k1 + 3. * k2) * dt)


class ExplicitRungeKutta3Kutta(ExplicitRungeKutta):
    """
    Kutta's three-stage, third-order ERK method
    """

    def __init__(self):
        super().__init__(name='Explicit Kutta RK3', order=3, stages=3)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 1.0 * dt, state + dt * (2.0 * k2 - k1))
        return StepOutput(solution_update=1. / 6. * (k1 + k3 + 4. * k2) * dt)


class ExplicitRungeKutta4Classical(ExplicitRungeKutta):
    """
    The classical four-stage, fourth-order ERK method
    """

    def __init__(self):
        super().__init__(name='Classical RK4', order=4, stages=4)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = rhs(t + dt, state + dt * k3)
        return StepOutput(solution_update=1. / 6. * (k1 + k4 + 2. * (k2 + k3)) * dt)


class AdaptiveERK21HeunEuler(AdaptiveExplicitRungeKutta):
    """
    The second-order Trapezoidal (Heun) method with a first-order Euler method for embedded error estimation
    """

    def __init__(self, norm_weighting=1.):
        super().__init__(name='Adaptive ERK 2(1) Heun-Euler',
                         order=2,
                         error_order=1,
                         stages=2,
                         norm_weighting=norm_weighting)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        k1 = rhs(t, state)
        k2 = rhs(t + dt, state + dt * k1)
        weighted_rhs_1 = k1
        weighted_rhs_2 = 0.5 * (k1 + k2)
        temporal_error = norm(dt * (weighted_rhs_2 - weighted_rhs_1) * self.norm_weighting, ord=self.norm_order)
        return StepOutput(solution_update=dt * weighted_rhs_2, temporal_error=temporal_error)


class AdaptiveERK54CashKarp(AdaptiveExplicitRungeKutta):
    """
    The fourth-order, five-stage adaptive ERK method of Cash and Karp
    """

    def __init__(self):
        super().__init__(name='Adaptive ERK 5(4) Cash-Karp',
                         order=5,
                         error_order=4,
                         stages=6)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        q = state
        r = rhs
        d = dt

        k1 = r(t, q)
        k2 = r(t + 0.2 * d, q + 0.2 * d * k1)
        k3 = r(t + 0.3 * d, q + 0.025 * d * (3. * k1 + 9. * k2))
        k4 = r(t + 0.6 * d, q + 0.1 * d * (3. * k1 - 9. * k2 + 12. * k3))
        k5 = r(t + 1.0 * d, q + 1. / 54. * d * (-11. * k1 + 135. * k2 - 140. * k3 + 70. * k4))
        k6 = r(t + 7. / 8. * d,
               q + 1. / 110592. * d * (3262. * k1 + 3.78e4 * k2 + 4.6e3 * k3 + 44275. * k4 + 6831. * k5))
        weighted_rhs_4 = 2825. / 27648. * k1 + 18575. / 48384. * k3 + 13525. / 55296. * k4 + 277. / 14336. * k5 + 0.25 * k6
        weighted_rhs_5 = 37. / 378. * k1 + 250. / 621. * k3 + 125. / 594. * k4 + 512. / 1771. * k6
        temporal_error = norm(d * (weighted_rhs_5 - weighted_rhs_4), ord=self.norm_order)
        return StepOutput(solution_update=d * weighted_rhs_5, temporal_error=temporal_error)


class GeneralAdaptiveExplicitRungeKutta(AdaptiveExplicitRungeKutta):
    """
    A general-purpose explicit Runge-Kutta method with embedded error estimation for adaptive time stepping

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper, e.g. 'Forward Euler' or 'Runge Kutta 4'
    order : int
        order of accuracy of the time stepping method
    A : np.ndarray
        Runge-Kutta stage coefficients for the method
    b : np.ndarray
        quadrature coefficients for the method
    bhat : np.ndarray
        embedded error estimation quadrature coefficients (default: None - not adaptive)
    """

    def __init__(self, name, order, A, b, bhat=None):
        # todo: should add checks that A is explicit and that the coefficients match the provided error
        self.A = copy(A)
        self.b = copy(b)
        self.bhat = copy(bhat) if bhat is not None else None
        super().__init__(name='General ERK: ' + name,
                         order=order,
                         error_order=order - 1,
                         stages=b.size)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        q = state
        r = rhs
        d = dt

        s = self._stages
        A = self.A
        b = self.b
        c = sum(A, axis=1)

        k = [r(t, q)]
        for i in range(1, s):
            stage_update = 0.
            for j in range(i):
                stage_update += A[i, j] * k[j]
            k.append(r(t + c[i] * d, q + d * stage_update))

        update = b[0] * k[0]
        error = zeros_like(update)
        if self.bhat is not None:
            error = (b[0] - self.bhat[0]) * k[0]
        for i in range(1, s):
            update += b[i] * k[i]
            if self.bhat is not None:
                error += (b[i] - self.bhat[i]) * k[i]

        return StepOutput(solution_update=d * update, temporal_error=norm(d * error, ord=self.norm_order))


class GeneralAdaptiveExplicitRungeKuttaMultipleEmbedded(AdaptiveExplicitRungeKutta):
    """
    A general-purpose explicit Runge-Kutta method with embedded error estimation for adaptive time stepping

    This class is used for advanced adaptive methods that utilize multiple embedded error estimates.

    **Constructor**:

    Parameters
    ----------
    name : str
        name of the TimeStepper, e.g. 'Forward Euler' or 'Runge Kutta 4'
    order : int
        order of accuracy of the time stepping method
    A : np.ndarray
        Runge-Kutta stage coefficients for the method
    b : np.ndarray
        quadrature coefficients for the method
    bhats : list[np.ndarray]
        list of embedded error estimation quadrature coefficients
    """

    def __init__(self, name, order, A, b, bhats):
        # todo: add checks that A is explicit and that the coefficients match the provided error
        self.A = copy(A)
        self.b = copy(b)
        self.bhats = bhats
        super().__init__(name='General ERK: ' + name,
                         order=order,
                         error_order=order - 1,
                         stages=b.size)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        q = state
        r = rhs
        d = dt

        s = self._stages
        A = self.A
        b = self.b
        c = sum(A, axis=1)
        nb = len(self.bhats)

        k = [r(t, q)]
        for i in range(1, s):
            stage_update = 0.
            for j in range(i):
                stage_update += A[i, j] * k[j]
            k.append(r(t + c[i] * d, q + d * stage_update))

        update = b[0] * k[0]
        error = zeros_like(update)
        extra_errors = [zeros_like(update)] * (nb - 1)
        if self.bhats is not None:
            error = (b[0] - self.bhats[0][0]) * k[0]
            for j in range(nb - 1):
                extra_errors[j] = (b[0] - self.bhats[j][0]) * k[0]
        for i in range(1, s):
            update += b[i] * k[i]
            if self.bhats is not None:
                error += (b[i] - self.bhats[0][i]) * k[i]
                for j in range(nb - 1):
                    extra_errors[j] = (b[0] - self.bhats[j][i]) * k[i]

        return StepOutput(solution_update=d * update,
                          temporal_error=norm(d * error, ord=self.norm_order),
                          extra_errors=[norm(d * e, ord=self.norm_order) for e in extra_errors])


class BackwardEuler(ImplicitTimeStepper):
    """
    The Backward Euler, or implicit Euler, or BDF-1, method

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    """

    def __init__(self, nonlinear_solver):
        super().__init__(name='Backward Euler',
                         order=1,
                         implicit_coefficient=1.,
                         nonlinear_solver=nonlinear_solver)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        state_n = copy(state)

        def residual(state_arg, existing_rhs=None, evaluate_new_rhs=True):
            rhs_val = rhs(t + dt, state_arg) if evaluate_new_rhs else existing_rhs
            return dt * rhs_val - (state_arg - state_n), rhs_val

        def linear_setup(state_arg):
            lhs_setup(t + dt, state_arg)

        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=rhs(t + dt, state))

        return StepOutput(solution_update=output.solution - state,
                          nonlinear_iter=output.iter,
                          linear_iter=output.liter,
                          nonlinear_converged=output.converged,
                          slow_nonlinear_convergence=output.slow_convergence,
                          projector_setups=output.projector_setups)


class BackwardEulerWithError(AdaptiveImplicitTimeStepper):
    """
    The Backward Euler, or implicit Euler, or BDF-1, method which uses the nonlinear solver to form an error estimate,
        allowing adaptivity as in the CVODE code (although without order adaptation - this class is first-order only)

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Backward Euler w/predictor-corrector error',
                         order=1,
                         implicit_coefficient=1.,
                         nonlinear_solver=nonlinear_solver,
                         norm_weighting=norm_weighting,
                         norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        state_n = copy(state)

        def residual(state_arg, existing_rhs=None, evaluate_new_rhs=True):
            rhs_val = rhs(t + dt, state_arg) if evaluate_new_rhs else existing_rhs
            return dt * rhs_val - (state_arg - state_n), rhs_val

        def linear_setup(state_arg):
            lhs_setup(t + dt, state_arg)

        max_nliter = self.nonlinear_solver.max_nonlinear_iter
        self.nonlinear_solver.max_nonlinear_iter = 1

        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=rhs(t + dt, state))
        state_predictor = output.solution
        nonlinear_iter = output.iter
        linear_iter = output.liter
        projector_setups = output.projector_setups

        self.nonlinear_solver.max_nonlinear_iter = max_nliter
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=output.rhs_at_converged)
        nonlinear_iter += output.iter
        linear_iter += output.liter
        projector_setups += output.projector_setups

        return StepOutput(solution_update=output.solution - state,
                          temporal_error=norm((output.solution - state_predictor) * self.norm_weighting,
                                              ord=self.norm_order),
                          nonlinear_iter=output.iter,
                          linear_iter=output.liter,
                          nonlinear_converged=output.converged,
                          slow_nonlinear_convergence=output.slow_convergence,
                          projector_setups=output.projector_setups)


class CrankNicolson(ImplicitTimeStepper):
    """
    The Crank-Nicolson, or implicit trapezoial, method

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    """

    def __init__(self, nonlinear_solver):
        super().__init__(name='Crank Nicolson',
                         order=2,
                         implicit_coefficient=0.5,
                         nonlinear_solver=nonlinear_solver)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        state_n = copy(state)
        rhs_n = rhs(t, state_n)

        def residual(state_arg, existing_rhs=None, evaluate_new_rhs=True):
            rhs_val = rhs(t + dt, state_arg) if evaluate_new_rhs else existing_rhs
            return 0.5 * dt * (rhs_val + rhs_n) - (state_arg - state_n), rhs_val

        def linear_setup(state_arg):
            lhs_setup(t + dt, state_arg)

        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=rhs(t + dt, state))

        return StepOutput(solution_update=output.solution - state,
                          nonlinear_iter=output.iter,
                          linear_iter=output.liter,
                          nonlinear_converged=output.converged,
                          slow_nonlinear_convergence=output.slow_convergence,
                          projector_setups=output.projector_setups)


class SDIRK22KForm(ImplicitTimeStepper):
    """
    A two-stage, second-order singly diagonally implicit method

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, nonlinear_solver):
        super().__init__(name='SDIRK22',
                         order=2,
                         implicit_coefficient=1. - 0.5 * sqrt(2.),
                         nonlinear_solver=nonlinear_solver)

        self.gamma = 1. - 0.5 * sqrt(2.)
        self.a21 = 0.5 * sqrt(2.)

        self.b1 = 0.5 * sqrt(2.)
        self.b2 = 1. - 0.5 * sqrt(2.)

        self.bvec = array([self.b1, self.b2])
        self.c = array([self.gamma, 1.])

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        nonlinear_iter = 0
        linear_iter = 0
        nonlinear_converged = True
        slow_nonlinear_convergence = False
        projector_setups = 0

        current_c_value = None
        prior_stage_k = None
        state_n = copy(state)

        g = self.gamma

        def residual(kval_arg, existing_rhs=None, evaluate_new_rhs=True):
            r = rhs(t + current_c_value * dt, state_n + dt * (g * kval_arg + prior_stage_k)) - kval_arg
            return r, r

        def linear_setup(kval_arg):
            lhs_setup(t + current_c_value * dt, state_n + dt * (g * kval_arg + prior_stage_k))

        # stage 1
        prior_stage_k = 0.
        current_c_value = self.c[0]
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=rhs(t + current_c_value * dt, state),
                                       initial_rhs=None)
        k1 = output.solution
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups
        prior_stage_k = self.a21 * k1
        current_c_value = self.c[1]

        # stage 2
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=k1,
                                       initial_rhs=None)
        k2 = output.solution
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        return StepOutput(solution_update=dt * (self.b1 * k1 + self.b2 * k2),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class SDIRK22(ImplicitTimeStepper):
    """
    A two-stage, second-order singly diagonally implicit method

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, nonlinear_solver):
        super().__init__(name='SDIRK22',
                         order=2,
                         implicit_coefficient=1. - 0.5 * sqrt(2.),
                         nonlinear_solver=nonlinear_solver)

        self.gamma = 1. - 0.5 * sqrt(2.)
        self.a21 = 0.5 * sqrt(2.)

        self.b1 = 0.5 * sqrt(2.)
        self.b2 = 1. - 0.5 * sqrt(2.)

        self.bvec = array([self.b1, self.b2])
        self.c = array([self.gamma, 1.])

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        nonlinear_iter = 0
        linear_iter = 0
        nonlinear_converged = True
        slow_nonlinear_convergence = False
        projector_setups = 0

        current_c_value = None
        prior_stage_k = None
        state_n = copy(state)

        def residual(state_arg, existing_rhs=None, evaluate_new_rhs=True):
            rhs_val = rhs(t + current_c_value * dt, state_arg) if evaluate_new_rhs else existing_rhs
            return dt * (self.gamma * rhs_val + prior_stage_k) - (state_arg - state_n), rhs_val

        def linear_setup(state_arg):
            lhs_setup(t + current_c_value * dt, state_arg)

        # stage 1
        prior_stage_k = 0.
        current_c_value = self.c[0]
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=rhs(t + current_c_value * dt, state))
        state = output.solution
        k1 = output.rhs_at_converged
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups
        prior_stage_k = self.a21 * k1
        current_c_value = self.c[1]

        # stage 2
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k1)
        k2 = output.rhs_at_converged
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        return StepOutput(solution_update=dt * (self.b1 * k1 + self.b2 * k2),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class ESDIRK64(AdaptiveImplicitTimeStepper):
    """
    The six-stage, fourth-order, second-stage-order, explicit singly diagonally implicit method of Kennedy and Carpenter

    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : np.ndarray or float
        how the temporal error estimate is weighted in its norm calculation (default: 1)
    norm_order : int or np.Inf
        the order of the norm used in the temporal error estimate (default: Inf)
    """

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='ESDIRK64',
                         order=4,
                         implicit_coefficient=0.25,
                         nonlinear_solver=nonlinear_solver,
                         norm_weighting=norm_weighting,
                         norm_order=norm_order)
        self.gamma = 0.25
        self.a21 = 0.25
        self.a31 = 8611. / 62500.
        self.a32 = -1743. / 31250.
        self.a41 = 5012029. / 34652500.
        self.a42 = -654441. / 2922500.
        self.a43 = 174375. / 388108.
        self.a51 = 15267082809. / 155376265600.
        self.a52 = -71443401. / 120774400.
        self.a53 = 730878875. / 902184768.
        self.a54 = 2285395. / 8070912.
        self.a61 = 82889. / 524892.
        self.a62 = 0.
        self.a63 = 15625. / 83664.
        self.a64 = 69875. / 102672.
        self.a65 = -2260. / 8211.

        self.b1 = self.a61
        self.b2 = self.a62
        self.b3 = self.a63
        self.b4 = self.a64
        self.b5 = self.a65
        self.b6 = self.gamma
        self.b1h = 4586570599. / 29645900160.
        self.b2h = 0.
        self.b3h = 178811875. / 945068544.
        self.b4h = 814220225. / 1159782912.
        self.b5h = -3700637. / 11593932.
        self.b6h = 61727. / 225920.

        self.bvec = array([self.b1, self.b2, self.b3, self.b4, self.b5, self.b6])
        self.bhvec = array([self.b1h, self.b2h, self.b3h, self.b4h, self.b5h, self.b6h])

        self.A = array([[0., 0., 0., 0., 0., 0.],
                        [self.a21, self.gamma, 0., 0., 0., 0.],
                        [self.a31, self.a32, self.gamma, 0., 0., 0.],
                        [self.a41, self.a42, self.a43, self.gamma, 0., 0.],
                        [self.a51, self.a52, self.a53, self.a54, self.gamma, 0.],
                        [self.a61, self.a62, self.a63, self.a64, self.a65, self.gamma]])
        self.c = sum(self.A, axis=1)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
        """
        Take a single step with this stepper method

        :param state: the current state
        :param t: the current time
        :param dt: the size of the time step
        :param rhs: the right-hand side of the ODE in the form f(t, y)
        :return: a StepOutput object
        """
        nonlinear_iter = 0
        linear_iter = 0
        nonlinear_converged = True
        slow_nonlinear_convergence = False
        projector_setups = 0

        current_c_value = None
        prior_stage_k = None
        state_n = copy(state)

        def residual(state_arg, existing_rhs=None, evaluate_new_rhs=True):
            rhs_val = rhs(t + current_c_value * dt, state_arg) if evaluate_new_rhs else existing_rhs
            return dt * (self.gamma * rhs_val + prior_stage_k) - (state_arg - state_n), rhs_val

        def linear_setup(state_arg):
            lhs_setup(t + current_c_value * dt, state_arg)

        # stage 1
        k1 = rhs(t, state)
        prior_stage_k = self.a21 * k1
        current_c_value = self.c[1]

        # stage 2
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k1)
        state = output.solution
        k2 = output.rhs_at_converged
        prior_stage_k = self.a32 * k2 + self.a31 * k1
        current_c_value = self.c[2]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 3
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k2)
        state = output.solution
        k3 = output.rhs_at_converged
        prior_stage_k = self.a43 * k3 + self.a42 * k2 + self.a41 * k1
        current_c_value = self.c[3]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 4
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k3)
        state = output.solution
        k4 = output.rhs_at_converged
        prior_stage_k = self.a54 * k4 + self.a53 * k3 + self.a52 * k2 + self.a51 * k1
        current_c_value = self.c[4]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 5
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k4)
        state = output.solution
        k5 = output.rhs_at_converged
        prior_stage_k = self.a65 * k5 + self.a64 * k4 + self.a63 * k3 + self.a62 * k2 + self.a61 * k1
        current_c_value = self.c[5]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 6
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k5)
        k6 = output.rhs_at_converged
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # accumulation
        dstate = dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 + self.b4 * k4 + self.b5 * k5 + self.b6 * k6)
        dstate_h = dt * (self.b1h * k1 + self.b2h * k2 + self.b3h * k3 + self.b4h * k4 + self.b5h * k5 + self.b6h * k6)

        return StepOutput(solution_update=dstate,
                          temporal_error=norm((dstate - dstate_h) * self.norm_weighting, ord=self.norm_order),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)
