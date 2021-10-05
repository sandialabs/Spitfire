

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://travis-ci.org/sandialabs/Spitfire.svg?branch=master)](https://travis-ci.com/sandialabs/Spitfire)
[![Read the Docs](https://readthedocs.org/projects/yt2mp3/badge/?version=latest)](https://spitfire.readthedocs.io/en/latest/?badge=latest)

Spitfire is a Python/C++ library for solving complex chemistry and reaction-diffusion problems. It is most often used to construct tabulated chemistry models for reacting flow simulations. It also solves canonical reactor models and provides efficient, extensible numerical time integration capabilities. Spitfire has been used and developed primarily at Sandia National Laboratories and the University of Utah.

- [Installing Spitfire](#installing-spitfire)
  * [Prerequisites](#prerequisites)
    + [Python and C++ Dependencies](#python-and-c-dependencies)
    + [`TabProps` for Presumed PDF Mixing Models](#tabprops-for-presumed-pdf-mixing-models)
  * [Building and Installing Spitfire](#building-and-installing-spitfire)
  * [Running the Tests](#running-the-tests)
  * [Building the Documentation](#building-the-documentation)

# Installing Spitfire

## Prerequisites

### Python and C++ Dependencies
Spitfire requires Python3 (tested with 3.6, 3.7) with the following packages:
- `NumPy`
- `SciPy`
- `matplotlib`
- `Cython`
- `Cantera`
- `sphinx`
- `NumPydoc`

We also highly recommend installing `jupyter`.

Conda provides the easiest method of installing Spitfire's Python dependencies, primarily because it can install the Cantera Python interface easily.
The lines below will install Spitfire's dependencies.
```
conda install numpy scipy matplotlib Cython sphinx numpydoc
conda install -c cantera cantera
```
Along with the optional `conda install -c anaconda jupyter`.

The pip package manager may also be used although this is more difficult because you'll have to install the Cantera Python interface yourself (see their [GitHub repository](https://github.com/Cantera/cantera) for guidance).
Before installing Cantera, install the packages noted above, most of which can be done with `pip3`.

Finally, Spitfire requires a C++11 compiler and the BLAS/LAPACK libraries, which are commonly available on many systems
and conda-based toolchains can provide these. Often simply installing NumPy, as already required, is sufficient.


### `TabProps` for Presumed PDF Mixing Models
Spitfire can leverage the [TabProps](https://gitlab.multiscale.utah.edu/common/TabProps/) code developed at the University of Utah
to provide presumed PDF mixing models. TabProps also provides arbitrary order piecewise Lagrange interpolants for structured data
in up to five dimensions. A Python interface may be built to enable these capabilities in Spitfire.
Without TabProps, Spitfire still provides fully featured reaction modeling capabilities,
so if you aren't interested in mixing models, installing TabProps is optional.

TabProps and its Python interface can be installed with a conda toolchain using the following commands.
Fortunately conda can install a version of boost for C++ dependency.
```
conda install -c anaconda cmake
conda install -c conda-forge boost-cpp
conda install -c conda-forge pybind11

git clone https://gitlab.multiscale.utah.edu/common/TabProps.git

cd TabProps
mkdir build
cd build
cmake .. \
     -DENABLE_PYTHON=ON \
     -DENABLE_MIXMDL=ON \
     -DTabProps_UTILS=ON \
     -DTabProps_PREPROCESSOR=OFF \
     -DTabProps_ENABLE_TESTING=ON \
     -DCMAKE_BUILD_TYPE=Release

make -j4 install
```
Note that `cmake` may struggle to find the Python interpreter of your conda environment.
If this happens (you might see permission issues or see `/usr/lib` in the python paths, etc.),
you can add these lines to the `cmake` call above.
Substitute `[anaconda-path]` with something like `/opt/anaconda3`, change `[env-name]` to
the name of your conda environment, and replace `[py_minor_version]` with 6 or 7 for python 3.6, 3.7.
```
     -DPYTHON_LIBRARY=[anaconda-path]/envs/[env-name]/lib/libpython3.[py_minor_version]m.a \
     -DPYTHON_INCLUDE_DIR=[anaconda-path]/envs/[env-name]/bin/python3.[py_minor_version]m \
     -DPYTHON_EXECUTABLE=[anaconda-path]/envs/[env-name]/bin/python3.[py_minor_version] \
```

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
