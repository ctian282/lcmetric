from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([Extension("clightcone_CIC", ["lightcone_CIC.pyx"],
                                       include_dirs=["/usr/include/healpix_cxx/"],
                                       extra_compile_args=["-O3","-fopenmp"],
                                       extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                                       language="c++",
                                       ),
                             Extension("cutils", ["utils.pyx"],
                                       extra_compile_args=["-O3","-fopenmp"],
                                       language="c++",
                                       )])


)

#setup(name='utils_C', ext_modules=cythonize('utils_C.pyx'),)
#setup(name='lightcone_it_C', ext_modules=cythonize('lightcone_it_C.pyx', language="c++"),)
