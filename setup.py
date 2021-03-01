from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        Extension("metric", ["metric.pyx"],
                  include_dirs=["/usr/include/healpix_cxx/"],
                  extra_compile_args=["-O3","-fopenmp",  "-std=c++14"],
                  extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                  language="c++",
                  ),
        Extension("lensing", ["lensing.pyx"],
                  include_dirs=["/usr/include/healpix_cxx/"],
                  extra_compile_args=["-O3","-fopenmp",  "-std=c++14"],
                  extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                  language="c++",
                  ),
        Extension("clightcone_CIC", ["lightcone_CIC.pyx"],
                  include_dirs=["/usr/include/healpix_cxx/"],
                  extra_compile_args=["-O3","-fopenmp",  "-std=c++14"],
                  extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                  language="c++",
                  ),
        Extension("cutils", ["utils.pyx"],
                  extra_compile_args=["-O3","-fopenmp", "-std=c++14"],
                  language="c++",
                  ),
        Extension("cgeodesic", ["geodesic.pyx"],
                  include_dirs=["/usr/include/healpix_cxx/"],
                  extra_compile_args=["-O3","-fopenmp", "-std=c++14"],
                  extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                  language="c++",
                  ),
        Extension("clcdensity", ["lcdensity.pyx"],
                  include_dirs=["/usr/include/healpix_cxx/"],
                  extra_compile_args=["-O3","-fopenmp", "-std=c++14"],
                  extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
                  language="c++",
                  )])


)

#setup(name='utils_C', ext_modules=cythonize('utils_C.pyx'),)
#setup(name='lightcone_it_C', ext_modules=cythonize('lightcone_it_C.pyx', language="c++"),)
