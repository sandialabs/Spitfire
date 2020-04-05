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


class TimeStepperBase:
    """Read-only base class for all time stepping methods

    **Constructor**: build a StepOutput object, specifying all data here (the object will be read-only)

    Parameters
    ----------
    name : str
        the name of the method
    order : int
        the order of the method
    is_adaptive : bool
        whether or not the method computes embedded error estimates to enable adaptive time-stepping
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    is_implicit : bool
        whether or not the method is implicit
    nonlinear_solver : int
        the nonlinear solver used by an implicit method at each time step
    implicit_coefficient : list[float]
        the coefficient in the implicit method that pre-multiplies the Jacobian matrix in the linear system
    """
    __slots__ = ['name',
                 'order',
                 'n_stages',
                 'is_adaptive',
                 'norm_weighting',
                 'norm_order',
                 'is_implicit',
                 'nonlinear_solver',
                 'implicit_coefficient']

    def __init__(self, name, order, **kwargs):
        self.name = name
        self.order = order
        self.n_stages = kwargs['n_stages'] if 'n_stages' in kwargs else 1
        self.is_adaptive = kwargs['is_adaptive'] if 'is_adaptive' in kwargs else False
        self.norm_weighting = kwargs['norm_weighting'] if 'norm_weighting' in kwargs else 1.
        self.norm_order = kwargs['norm_order'] if 'norm_order' in kwargs else Inf
        self.is_implicit = kwargs['is_implicit'] if 'is_implicit' in kwargs else False
        self.nonlinear_solver = kwargs['nonlinear_solver'] if 'nonlinear_solver' in kwargs else None
        self.implicit_coefficient = kwargs['implicit_coefficient'] if 'implicit_coefficient' in kwargs else None

    def norm(self, x):
        return norm(x * self.norm_weighting, ord=self.norm_order)


class StepOutput:
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
        the number of times the linear system was set up (e.g. Jacobian evaluation-factorization)
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


class ForwardEulerS1P1(TimeStepperBase):
    def __init__(self):
        super().__init__(name='forward Euler', order=1, n_stages=1)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        return StepOutput(solution_update=dt * rhs(t, state))


class ExpMidpointS2P2(TimeStepperBase):
    def __init__(self):
        super().__init__(name='ERK2 midpoint', order=2, n_stages=2)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        return StepOutput(solution_update=rhs(t + 0.5 * dt, state + 0.5 * dt * rhs(t, state)) * dt)


class ExpRalstonS2P2(TimeStepperBase):
    def __init__(self):
        super().__init__(name='ERK2 Ralston', order=2, n_stages=2)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        k1 = rhs(t, state)
        return StepOutput(solution_update=0.25 * (k1 + 3. * rhs(t + 2. / 3. * dt, state + 2. / 3. * dt * k1)) * dt)


class ExpTrapezoidalS2P2Q1(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, norm_weighting=1., norm_order=Inf):
        super().__init__(name='ERK2 trapezoidal', order=2, n_stages=2,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        k1 = rhs(t, state)
        r2 = 0.5 * (rhs(t + dt, state + dt * k1) + k1)
        return StepOutput(solution_update=dt * r2, temporal_error=dt * self.norm(r2 - k1))


class RK3KuttaS3P3(TimeStepperBase):
    def __init__(self):
        super().__init__(name='ERK3 Kutta', order=3, n_stages=3)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 1.0 * dt, state + dt * (2.0 * k2 - k1))
        return StepOutput(solution_update=1. / 6. * (k1 + k3 + 4. * k2) * dt)


class BogackiShampineS4P3Q2(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, norm_weighting=1., norm_order=Inf):
        super().__init__(name='ERK4 Bogacki-Shampine', order=3, n_stages=4,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 0.75 * dt, state + 0.75 * dt * k2)
        r1 = 1. / 9. * (2. * k1 + 3. * k2 + 4. * k3)
        k4 = rhs(t + dt, state + dt * r1)
        r2 = (7. * k1 + 6. * k2 + 8. * k3 + 3. * k4) / 24.
        return StepOutput(solution_update=dt * r1, temporal_error=dt * self.norm(r2 - r1))


class RK4ClassicalS4P4(TimeStepperBase):
    def __init__(self):
        super().__init__(name='ERK4 classical', order=4, n_stages=4)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = rhs(t + dt, state + dt * k3)
        return StepOutput(solution_update=1. / 6. * (k1 + k4 + 2. * (k2 + k3)) * dt)


class CashKarpS6P5Q4(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Cash/Karp RK5', order=5, n_stages=6,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
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
        r4 = 2825. / 27648. * k1 + 18575. / 48384. * k3 + 13525. / 55296. * k4 + 277. / 14336. * k5 + 0.25 * k6
        r5 = 37. / 378. * k1 + 250. / 621. * k3 + 125. / 594. * k4 + 512. / 1771. * k6
        return StepOutput(solution_update=d * r5, temporal_error=d * self.norm(r5 - r4))


class ZonneveldS5P4Q3(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Zonneveld RK4', order=4, n_stages=5,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        q = state
        r = rhs
        d = dt

        k1 = r(t, q)
        k2 = r(t + 0.5 * d, q + 0.5 * d * k1)
        k3 = r(t + 0.5 * d, q + 0.5 * d * k2)
        k4 = r(t + d, q + d * k3)
        k5 = r(t + 0.75 * d, q + 1. / 32. * d * (5. * k1 + 7. * k2 + 13. * k3 - k4))
        r1 = (k1 + k4 + 2. * (k2 + k3)) / 6.
        r2 = (-3. * k1 + 14. * (k2 + k3) + 13. * k4 - 32. * k5) / 6.
        return StepOutput(solution_update=d * r1, temporal_error=d * self.norm(r1 - r2))


class ExpKennedyCarpetnerS6P4Q3(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Kennedy/Carpenter explicit RK4', order=4, n_stages=6,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        q = state
        r = rhs
        d = dt

        k1 = r(t, q)
        k2 = r(t + 0.5 * d, q + 0.5 * d * k1)
        k3 = r(t + 83. / 250. * d, q + d / 62500. * (13861. * k1 + 6889. * k2))
        k4 = r(t + 31. / 50. * d, q + d * (9408046702089. / 11113171139209. * k3 -
                                           116923316275. / 2393684061468. * k1 -
                                           2731218467317. / 15368042101831. * k2))
        k5 = r(t + 17. / 20. * d, q + d * (-451086348788. / 2902428689909. * k1 +
                                           -2682348792572. / 7519795681897. * k2 +
                                           12662868775082. / 11960479115383. * k3 +
                                           3355817975965. / 11060851509271. * k4))
        k6 = r(t + d, q + d * (647845179188. / 3216320057751. * k1 +
                               73281519250. / 8382639484533. * k2 +
                               552539513391. / 3454668386233. * k3 +
                               3354512671639. / 8306763924573. * k4 +
                               4040. / 17871. * k5))
        r1 = (82889. / 524892. * k1 +
              15625. / 83664. * k3 +
              69875. / 102672. * k4 -
              2260. / 8211. * k5 +
              0.25 * k6)
        r2 = (4586570599. / 29645900160. * k1 +
              178811875. / 945068544. * k3 +
              814220225. / 1159782912. * k4 -
              3700637. / 11593932. * k5 +
              61727. / 225920. * k6)
        return StepOutput(solution_update=d * r1, temporal_error=d * self.norm(r1 - r2))


class BackwardEulerS1P1Q1(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='backward Euler', order=1, n_stages=1,
                         is_implicit=True, implicit_coefficient=1., nonlinear_solver=nonlinear_solver,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
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
                          temporal_error=self.norm(output.solution - state_predictor),
                          nonlinear_iter=output.iter,
                          linear_iter=output.liter,
                          nonlinear_converged=output.converged,
                          slow_nonlinear_convergence=output.slow_convergence,
                          projector_setups=output.projector_setups)


class CrankNicolsonS2P2(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    """

    def __init__(self, nonlinear_solver):
        super().__init__(name='Crank Nicolson', order=2, n_stages=2,
                         is_implicit=True, implicit_coefficient=0.5, nonlinear_solver=nonlinear_solver)

    def single_step(self, state, t, dt, rhs, lhs_setup, lhs_solve, *args, **kwargs):
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


class KennedyCarpenterS6P4Q3(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """
    __slots__ = ['gamma', 'a21', 'a31', 'a32', 'a41', 'a42', 'a43', 'a51', 'a52', 'a53', 'a54',
                 'a61', 'a62', 'a63', 'a64', 'a65',
                 'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                 'b1h', 'b2h', 'b3h', 'b4h', 'b5h', 'b6h', 'A', 'c']

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Kennedy/Carpenter ESDIRK64', order=4, n_stages=6,
                         is_implicit=True, implicit_coefficient=0.25, nonlinear_solver=nonlinear_solver,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)
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
                          temporal_error=self.norm(dstate - dstate_h),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class KvaernoS4P3Q2(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """
    __slots__ = ['gamma', 'a21', 'a31', 'a32', 'a41', 'a42', 'a43',
                 'b1', 'b2', 'b3', 'b4',
                 'b1h', 'b2h', 'b3h', 'b4h', 'A', 'c']

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Kvaerno ESDIRK43', order=3, n_stages=4,
                         is_implicit=True, implicit_coefficient=0.4358665215, nonlinear_solver=nonlinear_solver,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)
        self.gamma = 0.4358665215
        self.a21 = 0.4358665215
        self.a31 = 0.490563388419108
        self.a32 = 0.073570090080892
        self.a41 = 0.308809969973036
        self.a42 = 1.490563388254106
        self.a43 = -1.235239879727145

        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.gamma
        self.b1h = 0.490563388419108
        self.b2h = 0.073570090080892
        self.b3h = 0.4358665215
        self.b4h = 0.

        self.A = array([[0., 0., 0., 0.],
                        [self.a21, self.gamma, 0., 0.],
                        [self.a31, self.a32, self.gamma, 0.],
                        [self.a41, self.a42, self.a43, self.gamma], ])
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
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # accumulation
        dstate = dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 + self.b4 * k4)
        dstate_h = dt * (self.b1h * k1 + self.b2h * k2 + self.b3h * k3 + self.b4h * k4)

        return StepOutput(solution_update=dstate,
                          temporal_error=self.norm(dstate - dstate_h),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class KennedyCarpenterS4P3Q2(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """
    __slots__ = ['gamma', 'a21', 'a31', 'a32', 'a41', 'a42', 'a43',
                 'b1', 'b2', 'b3', 'b4',
                 'b1h', 'b2h', 'b3h', 'b4h', 'A', 'c']

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Kennedy/Carpenter ESDIRK43', order=3, n_stages=4,
                         is_implicit=True, implicit_coefficient=1767732205903. / 4055673282236.,
                         nonlinear_solver=nonlinear_solver,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)
        self.gamma = 1767732205903. / 4055673282236.
        self.a21 = self.gamma
        self.a31 = 2746238789719. / 10658868560708.
        self.a32 = -640167445237. / 6845629431997.
        self.a41 = 1471266399579. / 7840856788654.
        self.a42 = -4482444167858. / 7529755066697.
        self.a43 = 11266239266428. / 11593286722821.

        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.gamma
        self.b1h = 2756255671327. / 12835298489170.
        self.b2h = -10771552573575. / 22201958757719.
        self.b3h = 9247589265047. / 10645013368117.
        self.b4h = 2193209047091. / 5459859503100.

        self.A = array([[0., 0., 0., 0.],
                        [self.a21, self.gamma, 0., 0.],
                        [self.a31, self.a32, self.gamma, 0.],
                        [self.a41, self.a42, self.a43, self.gamma], ])
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
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # accumulation
        dstate = dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 + self.b4 * k4)
        dstate_h = dt * (self.b1h * k1 + self.b2h * k2 + self.b3h * k3 + self.b4h * k4)

        return StepOutput(solution_update=dstate,
                          temporal_error=self.norm(dstate - dstate_h),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class KennedyCarpenterS8P5Q4(TimeStepperBase):
    """
    **Constructor**:

    Parameters
    ----------
    nonlinear_solver : spitfire.time.nonlinear.NonlinearSolver
        the solver used in each implicit stage
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """
    __slots__ = ['gamma', 'a21', 'a31', 'a32', 'a41', 'a42', 'a43', 'a51', 'a52', 'a53', 'a54',
                 'a61', 'a62', 'a63', 'a64', 'a65',
                 'a71', 'a72', 'a73', 'a74', 'a75', 'a76',
                 'a81', 'a82', 'a83', 'a84', 'a85', 'a86', 'a87',
                 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8',
                 'b1h', 'b2h', 'b3h', 'b4h', 'b5h', 'b6h', 'b7h', 'b8h', 'A', 'c']

    def __init__(self, nonlinear_solver, norm_weighting=1., norm_order=Inf):
        super().__init__(name='Kennedy/Carpenter ESDIRK85', order=5, n_stages=8,
                         is_implicit=True, implicit_coefficient=41. / 200., nonlinear_solver=nonlinear_solver,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)
        self.gamma = 41. / 200.
        self.a21 = 41. / 200.
        self.a31 = 41. / 400.
        self.a32 = -567603406766. / 11931857230679.
        self.a41 = 683785636431. / 9252920307686.
        self.a42 = 0.
        self.a43 = -110385047103. / 1367015193373.
        self.a51 = 3016520224154. / 10081342136671.
        self.a52 = 0.
        self.a53 = 30586259806659. / 12414158314087.
        self.a54 = -22760509404356. / 11113319521817.
        self.a61 = 218866479029. / 1489978393911.
        self.a62 = 0.
        self.a63 = 638256894668. / 5436446318841.
        self.a64 = -1179710474555. / 5321154724896.
        self.a65 = -60928119172. / 8023461067671.
        self.a71 = 1020004230633. / 5715676835656.
        self.a72 = 0.
        self.a73 = 25762820946817. / 25263940353407.
        self.a74 = -2161375909145. / 9755907335909.
        self.a75 = -211217309593. / 5846859502534.
        self.a76 = -4269925059573. / 7827059040749.
        self.a81 = -872700587467. / 9133579230613.
        self.a82 = 0.
        self.a83 = 0.
        self.a84 = 22348218063261. / 9555858737531.
        self.a85 = -1143369518992. / 8141816002931.
        self.a86 = -39379526789629. / 19018526304540.
        self.a87 = 32727382324388. / 42900044865799.

        self.b1 = self.a81
        self.b2 = self.a82
        self.b3 = self.a83
        self.b4 = self.a84
        self.b5 = self.a85
        self.b6 = self.a86
        self.b7 = self.a87
        self.b8 = self.gamma
        self.b1h = -975461918565. / 9796059967033.
        self.b2h = 0.
        self.b3h = 0.
        self.b4h = 78070527104295. / 32432590147079.
        self.b5h = -548382580838. / 3424219808633.
        self.b6h = -33438840321285. / 15594753105479.
        self.b7h = 3629800801594. / 4656183773603.
        self.b8h = 4035322873751. / 18575991585200.

        self.A = array([[0., 0., 0., 0., 0., 0., 0., 0.],
                        [self.a21, self.gamma, 0., 0., 0., 0., 0., 0.],
                        [self.a31, self.a32, self.gamma, 0., 0., 0., 0., 0.],
                        [self.a41, self.a42, self.a43, self.gamma, 0., 0., 0., 0.],
                        [self.a51, self.a52, self.a53, self.a54, self.gamma, 0., 0., 0.],
                        [self.a61, self.a62, self.a63, self.a64, self.a65, self.gamma, 0., 0.],
                        [self.a71, self.a72, self.a73, self.a74, self.a75, self.a76, self.gamma, 0.],
                        [self.a81, self.a82, self.a83, self.a84, self.a85, self.a86, self.a87, self.gamma]])
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
        prior_stage_k = self.a76 * k6 + self.a75 * k5 + self.a74 * k4 + self.a73 * k3 + self.a72 * k2 + self.a71 * k1
        current_c_value = self.c[6]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 7
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k6)
        k7 = output.rhs_at_converged
        prior_stage_k = self.a87 * k7 + self.a86 * k6 + self.a85 * k5 + self.a84 * k4 + \
                        self.a83 * k3 + self.a82 * k2 + self.a81 * k1
        current_c_value = self.c[7]
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # stage 8
        output = self.nonlinear_solver(residual_method=residual,
                                       setup_method=linear_setup,
                                       solve_method=lhs_solve,
                                       initial_guess=state,
                                       initial_rhs=k7)
        k8 = output.rhs_at_converged
        nonlinear_iter += output.iter
        linear_iter += output.liter
        nonlinear_converged = nonlinear_converged and output.converged
        slow_nonlinear_convergence = slow_nonlinear_convergence or output.slow_convergence
        projector_setups += output.projector_setups

        # accumulation
        dstate = dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3 +
                       self.b4 * k4 + self.b5 * k5 + self.b6 * k6 + self.b7 * k7 + self.b8 * k8)
        dstate_h = dt * (self.b1h * k1 + self.b2h * k2 + self.b3h * k3 +
                         self.b4h * k4 + self.b5h * k5 + self.b6h * k6 + self.b7h * k7 + self.b8h * k8)

        return StepOutput(solution_update=dstate,
                          temporal_error=self.norm(dstate - dstate_h),
                          nonlinear_iter=nonlinear_iter,
                          linear_iter=linear_iter,
                          nonlinear_converged=nonlinear_converged,
                          slow_nonlinear_convergence=slow_nonlinear_convergence,
                          projector_setups=projector_setups)


class GeneralAdaptiveERK(TimeStepperBase):
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
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, name, order, A, b, bhat=None, norm_weighting=1., norm_order=Inf):
        # todo: should add checks that A is explicit and that the coefficients match the provided error
        self.A = copy(A)
        self.b = copy(b)
        self.bhat = copy(bhat) if bhat is not None else None
        super().__init__(name='General ERK: ' + name, order=order, n_stages=b.size,
                         is_adaptive=bhat is not None, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        q = state
        r = rhs
        d = dt

        s = self.n_stages
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

        return StepOutput(solution_update=d * update, temporal_error=d * self.norm(error))


class GeneralAdaptiveERKMultipleEmbedded(TimeStepperBase):
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
    norm_weighting : float or np.ndarray the size of the state vector
        multiplies the embedded error estimate prior to computing the norm (default: 1.)
    norm_order : int or np.Inf
        order of the norm of the error estimate (default: np.Inf)
    """

    def __init__(self, name, order, A, b, bhats, norm_weighting=1., norm_order=Inf):
        # todo: add checks that A is explicit and that the coefficients match the provided error
        self.A = copy(A)
        self.b = copy(b)
        self.bhats = bhats
        super().__init__(name='General ERK: ' + name, order=order, n_stages=b.size,
                         is_adaptive=True, norm_weighting=norm_weighting, norm_order=norm_order)

    def single_step(self, state, t, dt, rhs, *args, **kwargs):
        q = state
        r = rhs
        d = dt

        s = self.n_stages
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
                          temporal_error=d * self.norm(error),
                          extra_errors=[d * self.norm(e) for e in extra_errors])
