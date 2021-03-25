

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/sandialabs/Spitfire.svg?branch=master)](https://travis-ci.org/sandialabs/Spitfire)
[![Read the Docs](https://readthedocs.org/projects/yt2mp3/badge/?version=latest)](https://spitfire.readthedocs.io/en/latest/?badge=latest)

Spitfire is a Python/C++ library for solving complex chemistry and reaction-diffusion problems. It is most often used to construct tabulated chemistry models for reacting flow simulations. It also solves canonical reactor models and provides efficient, extensible numerical time integration capabilities. Spitfire has been used and developed primarily at Sandia National Laboratories and the University of Utah.

- [Installing Spitfire](#installing-spitfire)
  * [Prerequisites](#prerequisites)
    + [Python Packages](#python-packages)
    + [C++ Dependencies](#c---dependencies)
    + [Prerequisite installation using Conda](#prerequisite-installation-using-conda)
    + [Prerequisite installation without Conda](#prerequisite-installation-without-conda)
  * [Building and Installing Spitfire](#building-and-installing-spitfire)
  * [Running the Tests](#running-the-tests)
  * [Building the Documentation](#building-the-documentation)

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
After installing the prerequisites, clone the Spitfire repository, `cd` to the base repository directory,
and run the following command.
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
python3 -m unittest discover -s tests
```

## Building the Documentation
First, be aware that static documentation for Spitfire is hosted by [Read the Docs](https://spitfire.readthedocs.io/en/latest/).
A number of demonstrations as well as some basic theory are available in the documentation.
Second, documenting multi-language software in scientific applications, especially when extensibility is an explicit aim, is hard!
Any questions, suggestions, or help you could provide would be appreciated greatly.
If you want your own copy of the docs, or if you're developing in Spitfire and want to make sure your new documentation looks good, you can simply run the following commands,
```
cd docs
make html
```
and then point your favorite web browser to the `build/html/index.html` file.
Sphinx enables other forms of documentation but the HTML has been our primary target.