Introduction
------------

Objectives & Applications
=========================
Spitfire is a Python/C++ code for scientific computing that has several objectives:

- Solve canonical combustion problems with complex chemical kinetics using advanced time integration and continuation techniques.
- Build tabulated chemistry models for use in reacting flow simulation.
- Design and rapidly prototype solvers and time integration techniques for general-purpose ordinary and partial differential equations.

Spitfire has been used by researchers at Sandia National Laboratories and the University of Utah for a number of applications:

- construct adiabatic and nonadiabatic flamelet (tabulated chemistry) libraries for simulation of single- and multi-phase combustion
- investigate the design of specialized embedded pairs of explicit Runge-Kutta methods and advanced adaptive time stepping techniques
- generate homogeneous reactor, non-premixed flamelet, and general reaction-diffusion datasets for the training of low-dimensional surrogate models of chemical kinetics
- perform fundamental studies of combustion in the MILD regime
- study chemical explosive modes and low-temperature oxidation pathways of complex fuels in non-premixed systems
- study the formulation of state vectors and analytical Jacobian matrices for combustion simulation

For tabulated chemistry and reactor-based studies of chemical kinetics,
Spitfire provides several useful layers ultimately stacked into high-level constructs for simpler use.
At the lowest level, Spitfire provides optimized functions (written in Griffon, Spitfire's internal C++ engine) for evaluating quantities such as thermodynamic state functions and chemical reaction rates,
with a focus on reaction mechanisms up to hundreds of species and thousands of reactions.
With these an advanced researcher could compose high-level algorithms of their choosing with exact control over every detail.
However, Spitfire also provides higher-level classes for homogeneous reactors (all combinations of isochoric/isobaric, closed/open, adiabatic/isothermal/diathermal)
and non-premixed flamelets.
These classes provide methods needed to perform time integration and compute steady states of such systems.
While the `Flamelet` class allows a user to build tabulated chemistry libraries on their own,
Spitfire has tabulation routines with simpler interfaces for common types of libraries.

For time-stepping problems Spitfire employs a general abstraction of time integration appropriate for a range of integration techniques, nonlinear solvers, and linear solvers.
Design of this solver stack is important to the efficient solution of reacting flow problems with complex chemistry, and the exploration of solver algorithms and study of complex combustion chemistry were Spitfire's original purposes.
Combining NumPy, SciPy, and Cython with an internal C++ code called Griffon yields an experience with the convenience and extensibility of Python and the performance of precompiled languages like C/C++.
In this paradigm Python is used to drive abstract numerical algorithms at a high level while Griffon and C/Fortran routines wrapped by NumPy/SciPy are used in performance critical code.


Access
======
Spitfire is a BSD(3) open-sourced code that is hosted publicly on Sandia National Laboratory's GitHub page: https://github.com/sandialabs/Spitfire

Static documentation for Spitfire is hosted by Read the Docs: https://spitfire.readthedocs.io/en/latest/

A number of demonstrations can be seen on the "Using Spitfire" portion of the readme: https://github.com/sandialabs/Spitfire#using-spitfire

Authors
=======
Mike Hansen (mahanse@sandia.gov) is Spitfire's primary author and point of contact.
Others who have contributed to Spitfire are Elizabeth Armstrong, James Sutherland, Josh McConnell, John Hewson, and Robert Knaus.
