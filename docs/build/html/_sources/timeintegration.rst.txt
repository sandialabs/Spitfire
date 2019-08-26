Time Integration
================

Methods for First-Order Ordinary Differential Equations
-------------------------------------------------------

Spitfire can solve general differential equations (DEs) that can be written in explicit ordinary differential equation (ODE) form,

.. math::
 \frac{\partial \boldsymbol{q}}{\partial t} = \boldsymbol{r}(t, \boldsymbol{q}),
 :label: general_explicit_ode

where :math:`\boldsymbol{q}=[q_1,q_2,\ldots]` is the vector of state variables
and :math:`\boldsymbol{r}=[r_1,r_2,\ldots]` is the vector of right-hand side functions.
Many systems of scientific and engineering interest fit into this form,
either as ODEs derived naturally as :eq:`general_explicit_ode` or as
partial differential equations (PDEs) in semi-discrete form (after spatial, but not temporal, discretization).
Spitfire does not currently support implicit differential equations or differential algebraic equation (DAE) systems
represented generally as :math:`\mathrm{M}\dot{\boldsymbol{q}}=\boldsymbol{r}(t,\boldsymbol{q})` for a possibly singular matrix :math:`\mathrm{M}`.

Spitfire provides a number of explicit and implicit numerical methods of solving DE systems in the form of :eq:`general_explicit_ode`.
All of these methods may be classified as one-step Runge-Kutta methods.
Spitfire does not yet support multi-step methods such as BDF or Adams methods, or general linear methods.
Additionally Spitfire's abstraction does not support fully-implicit Runge-Kutta methods (singly-diagonally implicit `is` supported).
Finally Spitfire does not (yet!) support implicit-explicit methods such as additive Runge-Kutta or operator splitting techniques.

The explicit methods provided are:

- Forward Euler: ``ForwardEuler``
- Midpoint method: ``ExplicitRungeKutta2Midpoint``
- Trapezoidal method: ``ExplicitRungeKutta2Trapezoid``
- Ralston's two-stage method: ``ExplicitRungeKutta2Ralston``
- Heun's three-stage method: ``ExplicitRungeKutta3Kutta``
- The 'classical' RK4 method: ``ExplicitRungeKutta4Classical``
- The Cash-Karp order 4 method: ``AdaptiveERK54CashKarp``
- A general-purpose explicit RK method that can use any explicit Butcher table provided by the user: ``GeneralAdaptiveExplicitRungeKutta``

Implicit methods that come with Spitfire are:

- Backward Euler (BDF1): ``BackwardEuler``
- Crank-Nicolson (implicit Trapezoidal): ``CrankNicolson``
- Midpoint method: ``ImplicitMidpoint``
- Two-stage, L-stable, order 2 method: ``SDIRK22``
- Six-stage, L-stable, order 4 method with stage order two: ``ESDIRK64``

Spitfire provides the above ODE methods 'out of the box' and also facilitates the use of additional methods defined by the user.
In addition to the above general-purpose methods,
Spitfire can do error-based adaptive time-stepping with the following methods:

- explicit: Euler-Trapezoidal order 2 method: ``AdaptiveERK21HeunEuler``
- explicit: Cash-Karp order 4 method: ``AdaptiveERK54CashKarp``
- implicit: backward Euler: ``BackwardEulerWithError``
- implicit: six-stage, L-stable, order 4 method: ``ESDIRK64``

With the time-steppers listed above, Spitfire provides several means of driving a simulation in time.
All simulations, whether with explicit or implicit methods, with a constant or adaptive time step, are driven by Spitfire's ``Governor`` class.
The governor manages the logging of information and in-situ post-processing of user data as the simulation proceeds.
It also manages the evaluation of the Jacobian/preconditioning matrix used in implicit methods, depending upon performance
of nonlinear and linear solvers in each implicit time step.


Spitfire's Abstraction of the Solver Stack
------------------------------------------
When ODEs such as :eq:`general_explicit_ode` are solved with implicit time integration methods, a nonlinear system of equations must be solved at each time step.
The nonlinear system can be written in terms of a nonlinear operator :math:`\boldsymbol{\mathcal{N}}`,

.. math::
 \boldsymbol{\mathcal{N}}(\boldsymbol{q}) = \boldsymbol{0}.
 :label: eqn: simple nlin

A corresponding approximate linear operator :math:`\widetilde{\mathrm{A}}` is required in solving an exact linear problem required by the nonlinear problem,

.. math::
 \widetilde{\mathrm{A}} = \bar{p}\widetilde{\boldsymbol{\mathcal{N}}_{\boldsymbol{q}}} - \mathcal{I} \quad \rightarrow \quad \mathrm{solving}\, \left[\bar{p}\boldsymbol{\mathcal{N}}_{\boldsymbol{q}} - \mathrm{I}\right]\boldsymbol{x}=\boldsymbol{b},
 :label: eqn: simple lin

where the prefactor :math:`\bar{p}=ph` consists of the temporal discretization coefficient :math:`p` and time step size :math:`h`, the identity operator :math:`\mathcal{I}`, and identity matrix :math:`\mathrm{I}`, and the :math:`\widetilde{\boldsymbol{\mathcal{N}}_{\boldsymbol{q}}}` operator, an approximation of the Jacobian matrix :math:`\boldsymbol{\mathcal{N}}_{\boldsymbol{q}}`.
Nonlinear solution procedures typically require the repeated action of the inverse of the :math:`\mathrm{A}` operator, which can often be optimized by breaking it up into a costly setup phase (*e.g.*, factorization, preconditioner computation) and cheaper solve phase (*e.g.*, back-solution after factorization) so that the setup is called once per solve while setup is called many times.
The linear problem is a subset of the nonlinear problem, which itself is a subset of each single time step (:math:`t^n\to t^{n+1}`), which is a subset of a time integration loop with possibly adaptive time stepping (varying :math:`h` in time).
These five pieces form the backbone of time integration with implicit methods.
In Spitfire these are viewed as the ``Governor`` (time loop), ``StepController`` (:math:`h` adaptation), ``TimeStepper`` (single step method), ``NonlinearSolver`` (solve :math:`\boldsymbol{\mathcal{N}}(\boldsymbol{q}) = \boldsymbol{0}`), and finally the ``setup`` and ``solve`` procedures for the linear solve (build the approximate linear operator's inverse and repeatedly apply it, respectively).

Using Explicit Methods
++++++++++++++++++++++

Note that when explicit methods are used to solve :eq:`general_explicit_ode`, things are simplified dramatically because only the ``Governor``, ``StepController``, and ``TimeStepper`` behavior is needed.
The use of explicit methods is demonstrated by several scripts in the `demo/time_integration` folder.
For example, the `ballistics.py` (and jupyter notebook version `ballistics-demo.ipynb`), `chemistry-abc.py`, `ecology.py`, and `exponential_decay.py` scripts use explicit methods to solve various specific cases of :eq:`general_explicit_ode`.
Using Spitfire is quite simple here, as the use of the governor simply requires the specification of a termination criterion::

    governor = Governor()
    governor.termination_criteria = FinalTime(final_time)

and then the call to the integration method, which is given a right-hand side function as in :eq:`general_explicit_ode`,
an initial condition, a step controller (or if the :math:`h` is constant, the value of :math:`h`),
and an instance of the ``TimeStepper`` method to use (this example is from `chemistry-abc.py`)::

    governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k_ab, k_bc),
                       initial_condition=c0,
                       controller=time_step_size,
                       method=ExplicitRungeKutta4Classical())

In order to save data from the simulation, we make a container and provide its ``save_data`` method to the governor before calling ``integrate``::

    data = SaveAllDataToList(initial_solution=c0)
    governor.custom_post_process_step = data.save_data

This lets us obtain the solution times and values as follows, for instance::

    t = data.t_list
    q0 = data.solution_list[:, 0]

To use an adaptive time stepping approach based on temporal error control, one can build a ``PIController`` instance and
use a method such as ``AdaptiveERK54CashKarp``.
The corresponding integration call would be::

    governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k_ab, k_bc),
                       initial_condition=c0,
                       controller=PIController(),
                       method=AdaptiveERK54CashKarp())

As a final note, many of the instances we've built for the step controller and time steppers can be built with
optional parameters (*e.g.* the desired target error for the step controller, or the first time step).
In many cases default values are mostly acceptable.
See the module documentation to learn about available parameters.


Using Implicit Methods
++++++++++++++++++++++

Implicit methods may actually be used nearly as easily as explicit methods in simple cases.
In the `exponential_decay.py` script, the Backward Euler method is used and its integrate call is quite simple::

    governor.integrate(right_hand_side=lambda t, y: right_hand_side(y, k),
                       initial_condition=c0,
                       controller=time_step_size,
                       method=BackwardEuler(SimpleNewtonSolver()))

The only distinction between this and the explicit methods is that the ``BackwardEuler`` instance is built with a
``SimpleNewtonSolver`` object for solving the nonlinear system.
This simplicity is present in this case because we are letting Spitfire use a default dense linear solver (LU factorization and back-solution with LAPACK)
and a finite difference approximation to the Jacobian matrix.
In cases where a dense solver is appropriate this is a convenient option that will work very generally.

However, the challenge in efficiently using implicit methods for large problems is that the dense linear solver and
finite differenced Jacobian will not scale well.
For problems like nonpremixed flamelets described in the combustion section, this strategy is completely impractical.
Even in cases like the homogeneous reactors (also in the combustion section), where LAPACK is used, the finite
difference approximation to the Jacobian is too expensive and scales poorly with problem size.
For these reasons Spitfire provides the option of customizing the linear solver details.

As discussed above, solution of the linear system can often be broken down into a ``setup`` phase and a ``solve`` phase.
The setup phase might involve evaluation and factorization of the Jacobian matrix or assembly of a precoditioning matrix (for a Krylov solver).
The solve phase might involve back-solution with a direct solver such as LU, sparse LU, or a specialized direct algorithm,
or it might use a Krylov method like GMRES, CG, BiCGStab, *etc.*, possibly in a matrix-free manner.
Spitfire builds the separation of the ``setup`` and ``solve`` phases into the abstraction.

For a simple example of an implicit method with a customized linear solver, see the `demo/time_integration/chemistry_abc_implicit.py` script.
In this script a ``ChemistryProblem`` class we use LAPACK LU factorization of the Jacobian matrix, computed from a closed analytical result.
A ``lhs_inverse_op`` is stored in the ``setup`` phase when we build the augmented Jacobian matrix, :math:`\bar{p}\boldsymbol{\mathcal{N}}_{\boldsymbol{q}} - \mathrm{I}`,
and it is then used when we compute the solution of the linear system given a residual argument.
Note that the ``setup`` method takes two arguments: the prefactor :math:`\bar{p}` and the state vector.
The prefactor, :math:`\bar{p}=ph`, is provided to this function when called by the ``Governor`` and/or ``NonlinearSolver`` and incorporates
the temporal discretization coefficient :math:`p` from the ``TimeStepper`` and time step :math:`h` from the ``Governor``.
The ``solve`` method then takes only the residual vector and produces the solution to the linear problem.
These methods are fed to the ``Governor``'s integrate method as the ``projector_setup`` and ``projector_solve`` arguments.
This demonstration shows how to use the LAPACK method as well as a simple (silly in this case) diagonal approximation of the Jacobian,
which is a common simple preconditioner for Krylov methods.
Careful inspection of Spitfire's output for those cases shows that using the diagonal approximation increases the required
nonlinear iteration count from 170 (LU of the full Jacobian) to 276, over a 60% increase (the linear solve does not provide a good direction for the Newton update).
The version that uses LAPACK is shown below::

    class ChemistryProblem(object):
        """
        This class defines the right-hand side, setup, and solve methods for implicit methods with custom linear solvers
        """

        def __init__(self, k_ab, k_bc):
            self._k_ab = k_ab
            self._k_bc = k_bc
            self._lhs_inverse_op = None
            self._identity_matrix = np.eye(3)

        def rhs(self, t, c):
            c_a = c[0]
            c_b = c[1]
            q_1 = self._k_ab * c_a
            q_2 = self._k_bc * c_a * c_b
            return np.array([-q_1 - q_2,
                             q_1 - q_2,
                             2. * q_2])

        def setup_lapack_lu(self, c, prefactor):
            c_a = c[0]
            c_b = c[1]
            dq1_da = self._k_ab
            dq1_db = 0.
            dq1_dc = 0.
            dq2_da = self._k_bc * c_b
            dq2_db = self._k_bc * c_a
            dq2_dc = 0.
            J = np.array([[-dq1_da - dq2_da, -dq1_db - dq2_db, -dq1_dc - dq2_dc],
                          [dq1_da - dq2_da, dq1_db - dq2_db, dq1_dc - dq2_dc],
                          [2. * dq2_da, 2. * dq2_db, 2. * dq2_dc]])

            linear_op = prefactor * J - self._identity_matrix

            self._lhs_inverse_op = lapack_lu_factor(linear_op)[:2]  # the [:2] part here is just an implementation detail of scipy's lapack wrapper

        def solve_lapack_lu(self, residual):
            return lapack_lu_solve(self._lhs_inverse_op[0],
                                   self._lhs_inverse_op[1],
                                   residual)[0], 1, True      # the , 1, True parts here are how many iterations were needed and success/failure of the solver

        ...

        governor.integrate(right_hand_side=problem.rhs,
                   initial_condition=c0,
                   controller=time_step_size,
                   method=ESDIRK64(SimpleNewtonSolver()),
                   projector_setup=problem.setup_lapack_lu,
                   projector_solve=problem.solve_lapack_lu)
