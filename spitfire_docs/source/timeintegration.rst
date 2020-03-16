Time Integration
================

Methods for First-Order Ordinary Differential Equations
-------------------------------------------------------

Spitfire can solve general explicit ordinary differential equations

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

Spitfire provides a number of explicit and implicit numerical methods of solving systems in the form of :eq:`general_explicit_ode`.
All of these methods may be classified as one-step Runge-Kutta methods.
Spitfire does not yet support multi-step methods such as BDF or Adams methods, or general linear methods.
Additionally Spitfire's abstraction does not support fully-implicit Runge-Kutta methods (singly-diagonally implicit `is` supported).
Finally Spitfire does not (yet!) support implicit-explicit methods such as additive Runge-Kutta or operator splitting techniques,
but this will hopefully be tackled soon.

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
- explicit: general-purpose explicit Runge-Kutta method: ``GeneralAdaptiveExplicitRungeKutta``
- implicit: backward Euler: ``BackwardEulerWithError``
- implicit: six-stage, L-stable, order 4 method: ``ESDIRK64``

With the time-steppers listed above, Spitfire provides several means of driving a simulation in time.
All simulations, whether with explicit or implicit methods, with a constant or adaptive time step, are driven by Spitfire's ``Governor`` class.
The governor manages the logging of information and in-situ post-processing of user data as the simulation proceeds.
It also manages the evaluation of the Jacobian/preconditioning matrix used in implicit methods, depending upon performance
of nonlinear and linear solvers in each implicit time step.
Termination of time integration is managed by the ``Governor`` too.


Spitfire's Abstraction of the Implicit Solver Stack
---------------------------------------------------
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
Nonlinear solution procedures typically require the repeated action of the inverse of the :math:`\mathrm{A}` operator, which can often be optimized by breaking it up into a costly setup phase (*e.g.*, factorization, preconditioner computation) and cheaper solve phase (*e.g.*, back-solution after factorization) so that the setup is called once per solve while solve is called many times.
The linear problem is a subset of the nonlinear problem, which itself is a subset of each single time step (:math:`t^n\to t^{n+1}`), which is a subset of a time integration loop with possibly adaptive time stepping (varying :math:`h` in time).
These five pieces form the backbone of time integration with implicit methods - this is referred to as the 'solver stack.'
In Spitfire the stack consists of the ``Governor`` (time loop), ``StepController`` (:math:`h` adaptation), ``TimeStepper`` (single step method), ``NonlinearSolver`` (solve :math:`\boldsymbol{\mathcal{N}}(\boldsymbol{q}) = \boldsymbol{0}`), and finally the ``setup`` and ``solve`` procedures for the linear solve (building the inverse of the approximate linear operator and repeatedly applying it, respectively).

Using Explicit Methods
++++++++++++++++++++++

Note that when explicit methods are used to solve :eq:`general_explicit_ode`, things are relatively simple because only the ``Governor``, ``StepController``, and ``TimeStepper`` are needed.
The use of explicit methods is demonstrated by several scripts in the ``spitfire_demo/time_integration/explicit`` folder.
The simplest demonstration script, ``exp_decay_explicit.py``, solves an exponential decay problem, :math:`\partial y/\partial t = k y`,
with the Forward Euler method and the classical fourth-order Runge-Kutta method.

In another demonstration, ``ballistics.py`` in the same folder, a trajectory of a cannonball launched from the ground is determined for several values of the drag coefficient.
As it is challenging to select a single value for the time step in this problem, fourth-order automatic error-controlled time stepping is used (with a PI controller).
A key distinction of this demonstration is the use of a custom, user-defined termination rule.
As we want to integrate only until the cannonball has landed,
we write a method ``object_has_landed(state, *args, **kwargs)`` that is ``True`` when time integration should stop.
It simply checks that the object's center is lower than its radius off of the ground and that it currently is falling to the ground
(otherwise the laungh point would be caught immediately).
This is wrapped by a ``CustomTermination`` class and provided to the ``Governor`` object as shown here::

    def object_has_landed(state, *args, **kwargs):
        vel_y = state[1]
        pos_y = state[3]
        return not (pos_y < r and vel_y < 0)

    governor.termination_criteria = CustomTermination(object_has_landed)

The ballistics demonstration script also shows how to turn off output during the simulation, with::

    governor.do_logging = False

Another example of explicit time integration is included, named ``chemistry_explicit.py``.
This integrates a system of simple chemical reactions with the classical RK4 method.
Unlike the first two examples, it shows how to control the frequency of output from Spitfire::

    governor.log_rate = 100

As a final note, many of the step controllers and time steppers can be built with
optional parameters (*e.g.* the desired target error for the step controller or the first time step).
In many cases the default values are acceptable.
See the module documentation to learn about available parameters.

Another example of explicit time integration is the ``exp_decay_rk_study.py`` script,
which uses Spitfire's general-purpose explicit Runge-Kutta solver for the exponential decay problem.
Several Runge-Kutta methods including an eigth-order scheme are created from scratch in the script and
then plugged in to a ``Governor`` object as if they were provided by Spitfire in the first place.


Using Implicit Methods
++++++++++++++++++++++

Implicit methods may actually be used nearly as easily as explicit methods in simple cases.
Several demonstrations can be found in the ``spitfire_demo/time_integration/implicit`` folder.
In the `exp_decay_implicit.py` script, the Backward Euler method with a simple Newton's method solver is used and its integrate call is quite simple::

    governor.integrate(right_hand_side=rhs,
                       initial_condition=y0,
                       controller=dt,
                       method=BackwardEuler(SimpleNewtonSolver()))

The only distinction between this and the explicit methods is that the ``BackwardEuler`` instance is built with a
``SimpleNewtonSolver`` object for solving the internal nonlinear system.
This simplicity is present in this case because we are letting Spitfire use a default dense linear solver (LU factorization and back-solution with LAPACK)
and a finite difference approximation to the Jacobian matrix.
In cases where a dense solver and approximate Jacobian matrix are appropriate this is the most convenient option.

However, the challenge in efficiently using implicit methods for large problems is that the dense linear solver and
finite difference Jacobian matrix will not scale well.
For problems like nonpremixed flamelets described in the combustion section, this strategy is completely impractical.
Even in cases like the homogeneous reactors (also in the combustion section), where LAPACK is used, the finite
difference approximation to the Jacobian is expensive and scales poorly with problem size.
For these reasons Spitfire provides the option of customizing the linear solver details.

As discussed above, solution of the linear system can often be broken down into a ``setup`` phase and a ``solve`` phase.
The setup phase might involve evaluation and factorization of the Jacobian matrix or assembly of a preconditioning matrix (for a Krylov solver).
The solve phase might involve back-solution with a direct solver such as LU, sparse LU, or a specialized direct algorithm,
or it might use a Krylov method like GMRES, CG, BiCGStab, *etc.*, possibly in a matrix-free manner.
Spitfire builds the separation of the ``setup`` and ``solve`` phases into the abstraction.

For a simple example of an implicit method with a customized linear solver, see the `chemistry_implicit.py` script.
In this script a ``ChemistryProblem`` class is made to facilitate the details of the solver stack.
In the first integration, we use LAPACK LU factorization of the dense Jacobian matrix computed from an exact expression.
An ``lhs_inverse_op`` variable is saved in the ``setup`` phase when we build and decompose the augmented Jacobian matrix, :math:`\bar{p}\boldsymbol{\mathcal{N}}_{\boldsymbol{q}} - \mathrm{I}`,
and it is then used when we compute the solution of the linear system given a residual argument.
Note that the ``setup`` method takes two arguments: the prefactor :math:`\bar{p}` and the state vector.
The prefactor, :math:`\bar{p}=ph`, is provided to this function when called by the ``Governor`` and/or ``NonlinearSolver`` and incorporates
the temporal discretization coefficient :math:`p` from the ``TimeStepper`` and time step :math:`h` from the ``Governor``.
The ``solve`` method then takes only the residual vector and produces the solution to the linear problem.
These methods are fed to the ``Governor``'s integrate method as the ``projector_setup`` and ``projector_solve`` arguments.

This demonstration shows how to use the LAPACK method as well as a simple diagonal approximation of the Jacobian,
which is a common simple preconditioner for Krylov methods.
Careful inspection of Spitfire's output for those cases shows that using the diagonal approximation increases the required
nonlinear iteration count from 170 (LU of the full Jacobian) to 276, over a 60% increase due to the fact that the inexact linear solver does not provide as accurate of a direction for the nonlinear update.
A final integration is performed with the GMRES method using the diagonal matrix as a preconditioner (although it is not a great preconditioner for this problem).
Preconditioned GMRES does outperform direct use of the diagonal approximation, requiring only 213 nonlinear iterations in this case.
Note that this demonstration also shows how to obtain solver diagnostics from the integration call.