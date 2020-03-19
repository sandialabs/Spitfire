Spitfire is a Python-C++ library for solving complex chemistry and reaction-diffusion problems used in constructing tabulated chemistry models, built atop a variety of methods for solving nonlinear systems and differential equations.

# Prerequisites

## Python Packages
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

## C++ Dependencies
Spitfire's C++ internals are built with CMake, and we require at least version 3.7 of CMake.
CMake is often readily available on many systems, but if not you may install it following instructions at the [CMake webpage](https://cmake.org/) or with the Conda package manager (see below).
Spitfire also requires a C++11-compliant compiler and the BLAS/LAPACK libraries.

## Prerequisite installation using Conda
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

## Prerequisite installation without Conda
The pip package manager may also be used, as can direct installs, although this is discouraged because you'll have to install the Cantera Python interface yourself (see their [GitHub repository](https://github.com/Cantera/cantera) for guidance).
Before installing Cantera, install the packages noted above, most of which can be done with `pip3`.

# Spitfire Installation
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

# Testing
Spitfire has many tests that verify correctness or regression of the code.
To run the tests, go to the base repo directory and enter `python3 -m unittest discover -s spitfire_test/`.

# Examples
Some demonstrations can be found in the `spitfire_demo` directory.
The regression tests also serve as demonstrations (just ignore the `unittest` code within).

# Documentation
To build HTML documentation, navigate to the `spitfire_docs` directory and run `make html`.
Open `spitfire_docs/build/html/index.html` in your favorite web browser (e.g., on Mac run `open spitfire_docs/build/html/index.html` or on Linux, `firefox spitfire_docs/build/html/index.html`).
As always, documentation is in progress and don't hesitate to make issues on the GitHub page to ask questions.
