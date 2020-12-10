import pyximport
pyximport.install(reload_support=True)

from distutils.core import setup
from Cython.Build import cythonize

setup(name='utils_C', ext_modules=cythonize('utils_C.pyx'),)
setup(name='lightcone_it_C', ext_modules=cythonize('lightcone_it_C.pyx', language="c++"),)
