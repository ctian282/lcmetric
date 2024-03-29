from setuptools import Extension, setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([
    Extension(
        "metric",
        ["cython/metric.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    Extension(
        "lensing",
        ["cython/lensing.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=[],
        language="c++",
    ),
    Extension(
        "clightcone_CIC",
        ["cython/lightcone_CIC.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp", "-lhealpix_cxx", "-lz", "-lgomp"],
        language="c++",
    ),
    Extension(
        "cutils",
        ["cython/utils.pyx"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp", "-lgomp"],
        language="c++",
    ),
    Extension(
        "cgeodesic",
        sources=["cython/geodesic.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp", "-lhealpix_cxx", "-lz", "-lgomp"],
        language="c++",
    ),
    Extension(
        "clcdensity",
        ["cython/lcdensity.pyx"],
        include_dirs=["/usr/include/healpix_cxx/"],
        extra_compile_args=["-O3", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp", "-lhealpix_cxx", "-lz", "-lgomp"],
        language="c++",
    )
],
                            language_level=3))
