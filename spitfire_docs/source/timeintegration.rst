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
All simulations, whether with explicit or implicit methods, with a constant or adaptive time step, are driven by Spitfire's ``odesolve`` method.
This method manages the logging of information and in-situ post-processing of user data as the simulation proceeds,
evaluation of the Jacobian/preconditioning matrix used in implicit methods, depending upon performance
of nonlinear and linear solvers in each implicit time step.


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
In Spitfire the stack consists of ``odesolve`` (time loop), ``StepController`` (:math:`h` adaptation), ``TimeStepper`` (single step method), ``NonlinearSolver`` (solve :math:`\boldsymbol{\mathcal{N}}(\boldsymbol{q}) = \boldsymbol{0}`), and finally the ``setup`` and ``solve`` procedures for the linear solve (building the inverse of the approximate linear operator and repeatedly applying it, respectively).
