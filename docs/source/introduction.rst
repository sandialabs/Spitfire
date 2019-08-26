Introduction
============

Objectives & Access
-------------------
Spitfire is a Python/C++ code for scientific computing that has several objectives:

- Solve combustion problems with complex chemical kinetics, namely homogeneous reactors and non-premixed flamelets, both steady and unsteady, with advanced time integration and continuation techniques.
- Design and rapidly prototype nonlinear solvers and time integration techniques for ordinary and partial differential equations.

Spitfire is hosted internally on the CEE GitLab server: https://cee-gitlab.sandia.gov/spitfirecodes/spitfire

Mike Hansen is the primary author of Spitfire.
Important contributors include Elizabeth Armstrong, James Sutherland, and Josh McConnell.

Applications
------------
Spitfire utilizes a general abstraction of time integration appropriate for a range of integration techniques, nonlinear solvers, and linear solvers.
Design of this solver stack is crucial to efficient solution of reacting flow problems with complex chemistry
and the exploration of these algorithms and study of combustion chemistry was Spitfire's original purpose.
Spitfire combines Python with a C++ code called Griffon in addition to NumPy, SciPy, and Cython to attain the convenience and extensibility of Python and the performance of C/C++.
Python drives processes at an abstract level while either the Griffon code contained in Spitfire or NumPy- and SciPy-wrapped C/Fortran routines are used in performance critical operations.
Spitfire has been used by researchers at Sandia National Laboratories and the University of Utah for a number of applications:

- study chemical explosive modes and low-temperature oxidation pathways of complex fuels in non-premixed systems
- study the formulation of state vectors and analytical Jacobian matrices for combustion simulation
- construct steady adiabatic and nonadiabatic flamelet libraries for simulation of single- and multi-phase combustion
- perform fundamental studies of combustion in the MILD regime
- generate homogeneous reactor, non-premixed flamelet, and general reaction-diffusion datasets for the training of low-dimensional surrogate models of complex chemical models
- investigate the design of specialized embedded pairs of explicit Runge-Kutta methods and advanced adaptive time stepping techniques
