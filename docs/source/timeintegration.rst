Time Integration
================

.. toctree::
    :maxdepth: 1
    :caption: Demonstrations:

    demo/time_integration/explicit/explicit_exponential_decay_simple
    demo/time_integration/explicit/explicit_exponential_decay_custom_methods
    demo/time_integration/explicit/adaptive_stepping_and_custom_termination
    demo/time_integration/explicit/customized_adaptive_stepping
    demo/time_integration/explicit/lid_driven_cavity_scalar_mixing
    demo/time_integration/implicit/implicit_exponential_decay_simple
    demo/time_integration/implicit/implicit_advection_diffusion_linear_solvers_advanced
    demo/time_integration/implicit/implicit_diffusion_reaction

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

Methods are named according to stage count (S#), order of accuracy (P#), and the order of accuracy of the embedded error estimator (Q#) if present, similarly to ARKODE's convention.
If the embedded error order is noted, the method can be used for adaptive time-stepping based on local temporal error control.
You can add new methods of your own, even for adaptive time stepping - demonstrations in `spitfire_demo/time_integration` show how.

The explicit methods are:

- Forward Euler: ``ForwardEulerS1P1``
- Midpoint method: ``ExpMidpointS2P2``
- Trapezoidal method: ``ExpTrapezoidalS2P2Q1``
- Ralston's two-stage method: ``ExpRalstonS2P2``
- Kutta's three-stage method: ``RK3KuttaS3P3``
- The 'classical' RK4 method: ``RK4ClassicalS4P4``
- The Bogacki-Shampine four-stage method: ``BogackiShampineS4P3Q2``
- A five-stage order 4 method of Zonneveld: ``ZonneveldS5P4Q3``
- The Cash-Karp order 4 method: ``AdaptiveERK54CashKarp``
- Kennedy & Carpenter's six-stage order 4 method: ``ExpKennedyCarpetnerS6P4Q3``
- A general-purpose explicit RK method that can use any explicit Butcher table provided by the user: ``GeneralAdaptiveERK``

The implicit methods are:

- Backward Euler (BDF1): ``BackwardEulerS1P1Q1``
- Crank-Nicolson (implicit Trapezoidal): ``CrankNicolsonS2P2``
- Four-stage, order 3 method: ``KennedyCarpenterS4P3Q2``
- Four-stage, order 3 method: ``KvaernoS4P3Q2``
- Six-stage, L-stable, order 4 method with stage order two: ``KennedyCarpenterS6P4Q3``
- Eight-stage, L-stable, order 5 method with stage order two: ``KennedyCarpenterS8P5Q4``

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

Python & Performance Optimality
-------------------------------
Composing the solver stack in Python makes it more easily extensible, but it brings performance optimality into doubt.
Similar questions arise in development of HPC codes with C++, where virtual functions are 'slow' (and have other issues).
The key factor in performance is the cost of evaluating the nonlinear residual and the various operations in the linear solve.
In solving large, one-dimensional diffusion-reaction problems (flamelets), for which Spitfire is most often used, the large majority of time is spent evaluating the residual and Jacobian matrix and solving the linear system.
Performance improvements have come from optimizing the evaluation code and leveraging adaptive time-steppers that evaluate and factorize (the "setup" phase) fewer Jacobian matrices.
Early on we prototyped moving the entire solver stack, specializg it for a single time integration method, step control strategy, nonlinear solver, and linear solver to Griffon (Spitfire's internal C++ engine).
On even the smallest practical flamelet problems, this made no real difference in the end-to-end runtime, and the use of Python in the solver stack is entirely justified.
Now, if you're solving a single exponential decay ODE, Spitfire will be terribly slow compared to an optimized compiled application.
You may not care about performance as it will still be pretty fast, but certainly there are many-query applications that might want every last bit of performance on relatively small problems.
An option there is to combine ensembles of ODEs into one large system with optimized residual and Jacobian evaluation and linear solvers.
Solving many systems at once scales down the Python overhead and puts performance optimization in the hands of the user.
As a serial, single-threaded code meant for typical CPU hardware and a general problem space, our perspective with Spitfire is simply that performance is good enough on very small problems,
and is the user's responsibility in large problems (in computing the residual, Jacobian, and linear solver).
Thus, we write abstract numerical algorithms in Spitfire with extensibility and algorithmic optimality in mind.

