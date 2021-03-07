from setuptools import Extension, setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([
    Extension(
        "metric",
        ["src/metric.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=[],
        language="c++",
    ),
    Extension(
        "lensing",
        ["src/lensing.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=[],
        language="c++",
    ),
    Extension(
        "clightcone_CIC",
        ["src/lightcone_CIC.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
        language="c++",
    ),
    Extension(
        "cutils",
        ["src/utils.pyx"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        language="c++",
    ),
    Extension(
        "cgeodesic",
        ["src/geodesic.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
        language="c++",
    ),
    Extension(
        "clcdensity",
        ["src/lcdensity.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-lhealpix_cxx", "-lcfitsio", "-lz"],
        language="c++",
    )
],
                            language_level=2))
