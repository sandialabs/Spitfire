Spitfire is a Python-C++ library for solving complex chemistry and reaction-diffusion problems used in constructing tabulated chemistry models, built atop a variety of methods for solving nonlinear systems and differential equations.

# Prerequisites

## Python Packages
Spitfire requires Python3 (developed and tested with version 3.6) and the following Python packages:
- `numpy`
- `scipy`
- `matplotlib`
- `Cython`
- `cantera`
- `sphinx`
- `sphinx-rtd-theme`
- `numpydoc`

## C++ Dependencies
Spitfire's C++ internals are built with CMake, and it requires at least version 3.7.
CMake is often readily available on many systems, but if not you may install it following instructions at the [CMake webpage](https://cmake.org/) or with the Conda package manager.
Spitfire also requires a C++11-compliant compiler and the BLAS/LAPACK libraries.

## Prerequisite installation using Conda
Conda provides the easiest method of installing Spitfire's Python dependencies, primarily because it can install the cantera Python interface.
It is probably best to make an environment for Spitfire.
After activating it, run the following commands.
```
conda install numpy scipy matplotlib Cython sphinx numpydoc
conda install -c anaconda cmake
conda install -c cantera cantera
conda install -c anaconda sphinx_rtd_theme
```

## Prerequisite installation using pip
The pip package manager may also be used, although this is more challenging because you'll have to install the cantera Python interface yourself.
We use the latest version of cantera on their [GitHub repository](https://github.com/Cantera/cantera).
Before installing cantera, run the following commands.
```
pip3.6 install --upgrade numpy
pip3.6 install --upgrade scipy
pip3.6 install --upgrade matplotlib
pip3.6 install --upgrade Cython
pip3.6 install --upgrade sphinx
pip3.6 install --upgrade sphinx-rtd-theme
pip3.6 install --upgrade numpydoc
```

# Spitfire Installation
After installing the Python dependencies above, clone the Spitfire repository and `cd` to the `spitfire` directory.
```
git clone https://cee-gitlab.sandia.gov/spitfirecodes/spitfire
cd spitfire
```

First, we will build Griffon, a C++ code inside Spitfire.
This requires us to make a build directory, say `griffon_build`, and configure, compile, and install Griffon.
Fortunately CMake handles most of this for us.

1. Make a build directory (it can be anywhere, and you'll need to know the path again soon):
```
mkdir griffon_build
cd griffon_build
```

2. Now we run CMake from the build directory and point it to the `spitfire/griffon` directory (relative to the `spitfire` directory we made when cloning the repo).
```
cmake ../spitfire/griffon
```

3.Now simply run the following command to compile and install Griffon.
```
make -j4 install
```

Now navigate back to the root directory (in our case simply up one directory).
```
cd ..
```

Now run the following commands to 'Cythonize' the Griffon C++ code and install Spitfire for use in Python.
```
python3 setup.py build_ext --inplace --griffon-build-dir=griffon_build
python3 setup.py install --griffon-build-dir=griffon_build
```

# Testing
Spitfire has unit tests that verify correctness of many parts of the code,
and regression tests that simply run higher-level capabilities that do not admit easy correctness testing but are important to maintain.

To run the unit tests, go to the base repo directory and enter 
`python3 -m unittest discover -s spitfire_test/unit`.

To run the regression tests, go to the base repo directory and enter 
`python3 -m unittest discover -s spitfire_test/regression -v`. If you observe a `_bootstrap.py` warning that's expected...

# Examples
Some demonstrations can be found in the `demo` directory.
The regression tests also serve as demonstrations (just ignore the `unittest` code within).

# Documentation
To build HTML documentation, navigate to the `docs` directory and run `make html`.
Open `docs/build/html/index.html` in your favorite web browser (e.g., on Mac run `open docs/build/html/index.html` or on Linux, `firefox docs/build/html/index.html`).
