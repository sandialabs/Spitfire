

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/sandialabs/Spitfire.svg?branch=master)](https://travis-ci.org/sandialabs/Spitfire)
[![Read the Docs](https://readthedocs.org/projects/yt2mp3/badge/?version=latest)](https://spitfire.readthedocs.io/en/latest/?badge=latest)

# Introduction

Spitfire is a Python/C++ library for solving complex chemistry and reaction-diffusion problems. It is most often used to construct tabulated chemistry models for reacting flow simulations. It also solves canonical reactor models and provides efficient, extensible numerical time integration capabilities. Spitfire has been used and developed primarily at Sandia National Laboratories and the University of Utah.

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
- [Using Spitfire](#using-spitfire)
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
- `NumPydoc`

We also highly recommend installing `jupyter` and `dash`.

### C++ Dependencies
Spitfire requires a C++11-compliant compiler and the BLAS/LAPACK libraries, which are commonly available on many systems
and conda-based toolchains can provide these. Often simply installing NumPy, as already required, is sufficient.

### Prerequisite Installation
Conda provides the easiest method of installing Spitfire's Python dependencies, primarily because it can install the Cantera Python interface easily.
It is probably best to make an environment for Spitfire.
At the moment, stick to Python 3.6 or 3.7, as it is unclear if Spitfire and its dependencies run properly on Python 3.8.
To make a Python 3.7 environment named `spitfire` enter
```
conda create -n spitfire python=3.7
```
and then to activate it:
```
conda activate spitfire
```

After activating your environment, run the following commands to install the prerequisites.
```
conda install numpy scipy matplotlib Cython sphinx numpydoc
conda install -c cantera cantera
```
Also recommended are the following optional packages:
```
conda install -c anaconda jupyter
```

The pip package manager may also be used although this is more difficult because you'll have to install the Cantera Python interface yourself (see their [GitHub repository](https://github.com/Cantera/cantera) for guidance).
Before installing Cantera, install the packages noted above, most of which can be done with `pip3`.

## Building and Installing Spitfire
After installing the prerequisites, clone the Spitfire repository and `cd` to the base repository directory.
```
git clone https://github.com/sandialabs/Spitfire.git
cd Spitfire
```
Run the following command to install Spitfire for use in Python.
```
python3 setup.py install
```
If you want to run tests and build the documentation yourself, an in-place build is also required:
```
python3 setup.py build_ext --inplace install
```

## Running the Tests
Spitfire has a number of tests that verify correctness or regression of the code.
After installing Spitfire or developing code it is a great idea to run these tests.
To do this, go to the base repository directory and enter the following command:
```
python3 -m unittest discover -s spitfire_test/
```

## Building the Documentation
First, be aware that static documentation for Spitfire is hosted by [Read the Docs](https://spitfire.readthedocs.io/en/latest/).
Second, documenting multi-language software in scientific applications, especially when extensibility is an explicit aim, is hard!
Any questions, suggestions, or help you could provide would be appreciated greatly.
If you want your own copy of the docs, or if you're developing in Spitfire and want to make sure your new documentation looks good, you can simply run the following commands,
```
cd docs
make html
```
and then point your favorite web browser to the `build/html/index.html` file.
Sphinx enables other forms of documentation but the HTML has been our primary target.


# Using Spitfire
These annotated Jupyter notebooks show many key features and capabilities of Spitfire, split in the three categories below.
They've been `nbconvert`ed to HTML files and the links use `nbviewer` hosted on `jupyter.org` to show typeset documents.

## General-Purpose Time Integration
- Explicit methods
    - [Basics of using explicit methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/explicit_exponential_decay_simple.html)
    - [User-defined time-stepping methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/explicit_exponential_decay_custom_methods.html)
    - [Adaptive time stepping and custom termination rules for cannonball trajectories](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/adaptive_stepping_and_custom_termination.html)
    - [Customized adaptive time-stepping methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/customized_adaptive_stepping.html)
    - [Solving a lid-driven cavity problem with adaptive time-stepping](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/explicit/lid_driven_cavity_scalar_mixing.html)
- Implicit methods
    - [Basics of using implicit methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/implicit/implicit_exponential_decay_simple.html)
    - [Advanced linear solvers for advection-diffusion](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/implicit/implicit_advection_diffusion_linear_solvers_advanced.html)
    - [Solving an 'explosive' diffusion-reaction problem](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/time_integration/implicit/implicit_diffusion_reaction.html)

## Homogeneous Reactors 
  - [Introduction to Cantera and Spitfire](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/thermochemistry_Cantera_Spitfire_griffon.html)
  - [Creation of a One-Step Reaction Mechanism and Ignition Comparison](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/one_step_heptane_ignition.html)
  - [Computing Ignition Delay Profiles for a Two-Stage Ignition Fuel](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/ignition_delay_NTC_DME.html)
  - [Time-dependent Flow Reactor with Periodic Ignition/Extinction](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/oscillating_ignition_extinction.html)
  - [Chemical Explosive Mode Analysis of Isothermal/Adiabatic Reactors](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/isothermal_reactors_with_mode_analysis.html)
  - [Multiple Steady States in n-heptane/air Mixtures](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/reactors/ignition_extinction_heptane.html)

## Non-premixed Flamelets and Tabulated Chemistry
  - [Introduction to Flamelets in Spitfire](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/introduction_to_flamelets.html)
  - [Introduction to the Tabulation Methods](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/high_level_tabulation_api_adiabatic.html)
  - [Tabulation Methods for Nonadiabatic Models](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/example_nonadiabatic_flamelets.html)
  - [Model Generation for a Methane Shear Layer](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/methane_shear_layer_tabulation.html)
  - [Transient Ignition with Advanced Time Integration Options](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/transient_ignition_stepper_details.html)
  - [Simple Rate Sensitivity Study with Flamelet Ignition](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/example_transient_flamelet_rate_sensitivity.html)
  - [Custom Tabulation for a 4D Coal Combustion Model](https://nbviewer.jupyter.org/github/sandialabs/Spitfire/blob/master/spitfire_demo/flamelet/example_coal_combustion_model.html)
  
