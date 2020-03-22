# Introduction

Spitfire is a Python/C++ library for solving complex chemistry and reaction-diffusion problems used in constructing tabulated chemistry models, built atop a variety of methods for solving nonlinear systems and differential equations.

Spitfire has several key objectives:
- Solve canonical combustion problems with complex chemical kinetics using advanced time integration and continuation techniques.
- Build tabulated chemistry models for use in reacting flow simulation.
- Design and rapidly prototype solvers and time integration techniques for general-purpose ordinary and partial differential equations.


Spitfire has been used by researchers at Sandia National Laboratories and the University of Utah for a number of applications:
- construct adiabatic and nonadiabatic flamelet libraries (tabulated chemistry) for simulation of single- and multi-phase combustion, 
  similarly to how the FlameMaster code has been used
- investigate the design of specialized embedded pairs of explicit Runge-Kutta methods and advanced adaptive time stepping techniques
- generate homogeneous reactor, non-premixed flamelet, and simple reaction-diffusion datasets for the training of low-dimensional surrogate models of chemical kinetics
- perform fundamental studies of combustion in the MILD regime
- study chemical explosive modes and low-temperature oxidation pathways of complex fuels in non-premixed systems
- study the formulation of state vectors and analytical Jacobian matrices for combustion simulation


# Contents
- [Introduction](#introduction)
- [Contents](#contents)
- [Installing Spitfire](#installing-spitfire)
  * [Prerequisites](#prerequisites)
    + [Python Packages](#python-packages)
    + [C++ Dependencies](#c---dependencies)
    + [Prerequisite installation using Conda](#prerequisite-installation-using-conda)
    + [Prerequisite installation without Conda](#prerequisite-installation-without-conda)
  * [Building and Installing Spitfire](#building-and-installing-spitfire)
  * [Running the Tests](#running-the-tests)
  * [Building the Documentation](#building-the-documentation)
- [Running Spitfire](#running-spitfire)
  * [General-Purpose Time Integration](#general-purpose-time-integration)
  * [Homogeneous Reactors](#homogeneous-reactors)
  * [Non-premixed Flamelets and Tabulated Chemistry](#non-premixed-flamelets-and-tabulated-chemistry)

# Installing Spitfire

## Prerequisites

### Python Packages
Spitfire requires Python3 (developed and tested with version 3.6) and the following Python packages:
- `NumPy`
- `SciPy`
- `matplotlib`
- `Cython`
- `Cantera`
- `sphinx`
- `sphinx-rtd-theme`
- `NumPydoc`

We also highly recommend installing `jupyter` and `dash`.

### C++ Dependencies
Spitfire's C++ internals are built with CMake, and we require at least version 3.7 of CMake.
CMake is often readily available on many systems, but if not you may install it following instructions at the [CMake webpage](https://cmake.org/) or with the Conda package manager (see below).
Spitfire also requires a C++11-compliant compiler and the BLAS/LAPACK libraries.

### Prerequisite installation using Conda
Conda provides the easiest method of installing Spitfire's Python dependencies, primarily because it can install the Cantera Python interface.
It is probably best to make an environment for Spitfire.
At the moment, stick to Python 3.6, as it is unclear if Spitfire and its dependencies run properly on Python 3.7.
To make an environment named `spitfire` which will use Python 3.6, enter
```
conda create -n spitfire python=3.6
```
and then to activate it:
```
conda activate spitfire
```

After activating your environment, run the following commands to install the prerequisites.
```
conda install numpy scipy matplotlib Cython sphinx numpydoc
conda install -c anaconda cmake
conda install -c cantera cantera
conda install -c anaconda sphinx_rtd_theme jupyter
conda install -c conda-forge dash
```

### Prerequisite installation without Conda
The pip package manager may also be used, as can direct installs, although this is discouraged because you'll have to install the Cantera Python interface yourself (see their [GitHub repository](https://github.com/Cantera/cantera) for guidance).
Before installing Cantera, install the packages noted above, most of which can be done with `pip3`.

## Building and Installing Spitfire
After installing the prerequisites, clone the Spitfire repository and `cd` to the `Spitfire` directory.
```
git clone https://github.com/sandialabs/Spitfire.git
cd Spitfire
```

First, we will build Griffon, a C++ code inside Spitfire.
This requires us to make a build directory, say `griffon_build`, and configure, compile, and install Griffon.
Fortunately CMake handles most of this for us.

If you're using Conda and created a new environment for Spitfire, be sure to have it activated at this point.

1. Make a build directory and navigate to it:
```
mkdir griffon_build
cd griffon_build
```

2. Next, run CMake from the build directory and point it to the `spitfire/griffon` directory (relative to the `Spitfire` directory we made when cloning the repo).
```
cmake ../spitfire/griffon
```

3. Next, run the following command to compile and install Griffon.
```
make -j4 install
```

4. Now navigate back to the root directory (in our case simply up one directory).
```
cd ..
```

5. Run the following commands to 'Cythonize' the Griffon C++ code and install Spitfire for use in Python.
```
python3 setup.py build_ext --inplace --griffon-build-dir=griffon_build
python3 setup.py install --griffon-build-dir=griffon_build
```

## Running the Tests
Spitfire has many tests that verify correctness or regression of the code.
After installing Spitfire or developing code it is a great idea to run these tests.
To do this, go to the base repo directory and enter `python3 -m unittest discover -s spitfire_test/`.
Python's `unittest` module allows 

## Building the Documentation
To build HTML documentation that provides some background theory and an API reference for the entire code, enter the following commands (from within the base `Spitfire` directory).
Documentation is always in progress and could always be improved - please don't hesitate to make issues on the GitHub page to ask questions or point out confusing parts.

```
cd spitfire_docs
make html
```

Then point your favorite web browser to the `spitfire_docs/build/html/index.html` file.
For Mac OS X you can simply run `open build/html/index.html` and on Linux you could run `firefox build/html/index.html`


# Running Spitfire

## General-Purpose Time Integration
- Explicit methods
    - [Basics of using explicit methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/explicit_exponential_decay_simple.html)
    - [User-defined time-stepping methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/explicit_exponential_decay_custom_methods.html)
    - [Adaptive time stepping and custom termination rules for cannonball trajectories](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/adaptive_stepping_and_custom_termination.html)
    - [Customized adaptive time-stepping methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/customized_adaptive_stepping.html)
- Implicit methods
    - [Basics of using implicit methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/implicit/implicit_exponential_decay_simple.html)
    - [Advanced linear solvers for advection-diffusion](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/implicit/implicit_advection_diffusion_linear_solvers_advanced.html)

## Homogeneous Reactors 
  - in progress... see the scripts and notebooks in the spitfire_demo/reactors directory

## Non-premixed Flamelets and Tabulated Chemistry
  - in progress... see the scripts and notebooks in the spitfire_demo/flamelet directory