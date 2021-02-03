# Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
# Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
#                       
# You should have received a copy of the 3-clause BSD License                                        
# along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
#                    
# Questions? Contact Mike Hansen (mahanse@sandia.gov)    

import os
from glob import glob
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from numpy import get_include as numpy_include
import platform


def readfile(filename):
    try:
        with open(filename) as f:
            return f.read()
    except:
        with open(filename, encoding='utf-8') as f:
            return f.read()


def get_compile_args():
    eca = ['-O3', '-g', '-std=c++11', '-Wno-error']
    if platform.system() == 'Darwin':
        eca += ['-stdlib=libc++']
    return eca


def get_lapack():
    is_mac = platform.system() == 'Darwin'
    if is_mac:
        lib = []
        extra = ['-framework', 'Accelerate', '-mmacosx-version-min=10.12']
    else:
        lib = ['blas', 'lapack']
        extra = []
    return lib, extra


def make_griffon_extension():
    lapack_lib, lapack_extra = get_lapack()
    return cythonize(Extension(name='spitfire.griffon.griffon',
                               sources=[os.path.join('spitfire', 'griffon', 'griffon.pyx')] + glob(
                                   os.path.join('spitfire', 'griffon', 'src') + '/*.cpp', recursive=False),
                               extra_compile_args=get_compile_args(),
                               include_dirs=[numpy_include(), os.path.join('spitfire', 'griffon', 'include')],
                               library_dirs=[os.path.join('spitfire', 'griffon')],
                               libraries=lapack_lib,
                               extra_link_args=lapack_extra,
                               language='c++'))


def print_info():
    print('-' * 80)
    print(f'- Done installing Spitfire!\n')
    print(f'- To test or build docs, an in-place build is required:\n')
    print(f'    python3 setup.py build_ext --inplace\n')
    print('-' * 80)
    print(f'- Run the tests:\n')
    print(f'    python3 -m unittest discover -s spitfire_test')
    print('-' * 80)
    print(f'- Build the docs:\n')
    print(f'    cd spitfire_docs')
    print(f'    make html')
    print(f'    open build/html/index in a browser')
    print('-' * 80)


setup(name='Spitfire',
      version=readfile('version'),
      author='Michael A. Hansen',
      author_email='mahanse@sandia.gov',
      license=readfile('license.md'),
      description=readfile('description_short'),
      long_description=readfile('readme.md'),
      url='https://github.com/sandialabs/Spitfire/',
      packages=['spitfire', 'spitfire.chemistry', 'spitfire.time', 'spitfire.griffon'],
      ext_modules=make_griffon_extension(),
      package_data={'spitfire.griffon': ['*.so']},
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: C++',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
      python_requires='>=3.6')

print_info()
