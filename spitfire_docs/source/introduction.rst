Introduction
============

Objectives & Applications
-------------------------
Spitfire is a Python/C++ code for scientific computing that has several objectives:

- Solve canonical combustion problems with complex chemical kinetics using advanced time integration and continuation techniques.
- Build tabulated chemistry models for use in reacting flow simulation.
- Design and rapidly prototype solvers and time integration techniques for general-purpose ordinary and partial differential equations.

Spitfire utilizes a general abstraction of time integration appropriate for a range of integration techniques, nonlinear solvers, and linear solvers.
Design of this solver stack is important to the efficient solution of reacting flow problems with complex chemistry.
The exploration of solver algorithms and study of complex combustion chemistry were Spitfire's original purposes.

Spitfire combines NumPy, SciPy, and Cython with an internal C++ code called Griffon to obtain the convenience and extensibility of Python with the performance of C/C++.
Python drives abstract numerical algorithms at a high level while Griffon code and C/Fortran routines wrapped by NumPy and SciPy are used in performance critical code.
Spitfire has been used by researchers at Sandia National Laboratories and the University of Utah for a number of applications:

- construct adiabatic and nonadiabatic flamelet (tabulated chemistry) libraries for simulation of single- and multi-phase combustion
- investigate the design of specialized embedded pairs of explicit Runge-Kutta methods and advanced adaptive time stepping techniques
- generate homogeneous reactor, non-premixed flamelet, and general reaction-diffusion datasets for the training of low-dimensional surrogate models of chemical kinetics
- perform fundamental studies of combustion in the MILD regime
- study chemical explosive modes and low-temperature oxidation pathways of complex fuels in non-premixed systems
- study the formulation of state vectors and analytical Jacobian matrices for combustion simulation

Access
------
Spitfire is a BSD(3) open-sourced code that is hosted publicly on Sandia's GitHub page: https://github.com/sandialabs/Spitfire

Authors
-------
Mike Hansen (mahanse@sandia.gov) is Spitfire's primary author and point of contact.
Others who have contributed to Spitfire are Elizabeth Armstrong, James Sutherland, Josh McConnell, John Hewson, and Robert Knaus.
