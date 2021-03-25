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


setup(name='Spitfire',
      version=readfile('version'),
      author='Michael A. Hansen',
      author_email='mahanse@sandia.gov',
      license=readfile('license.md'),
      description=readfile('description_short'),
      long_description=readfile('readme.md'),
      url='https://github.com/sandialabs/Spitfire/',
      package_dir={'': 'src'},
      packages=['spitfire', 'spitfire.chemistry', 'spitfire.time', 'spitfire.griffon'],
      ext_modules=cythonize(Extension(name='spitfire.griffon.griffon',
                                      sources=[os.path.join('src', 'spitfire', 'griffon', 'griffon.pyx')] + glob(
                                          os.path.join('src', 'spitfire', 'griffon', 'src') + '/*.cpp', recursive=False),
                                      extra_compile_args=['-O3', '-g', '-std=c++11', '-Wno-error'] + [
                                          '-stdlib=libc++'] if platform.system() == 'Darwin' else [],
                                      include_dirs=[numpy_include(), os.path.join('src', 'spitfire', 'griffon', 'include')],
                                      library_dirs=[os.path.join('src', 'spitfire', 'griffon')],
                                      libraries=['blas', 'lapack'],
                                      language='c++')),
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

print(f"""
- Done installing Spitfire!
{'-' * 80}
- To test or build docs, an in-place build is also required:
    python3 setup.py build_ext --inplace
{'-' * 80}
- Run the tests:
    python3 -m unittest discover -s tests
{'-' * 80}
- Build the docs:
    cd docs
    make html
    open build/html/index.html in a browser
{'-' * 80}
- Update docs from Jupyter demos
    cd docs/source/demo
    find . -name *ipynb | xargs jupyter nbconvert --to rst
{'-' * 80}
""")
