# distutils: sources =
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

import numpy as npy
import healpy as hp

import lcmetric.utils as ut

cimport numpy as npy

cimport cgeodesic
from libcpp cimport bool


ctypedef long long idx_t
ctypedef double real_t

cdef class Geodesic:
    cdef _Geodesic[real_t, idx_t] *c_geo

    cdef bool ray_tracing

    cdef npy.ndarray Phi
    cdef npy.ndarray lm

    cdef double init_r, final_r

    cdef int NR, NSIDE, lmax

    def __cinit__(self, real_t [:,::1] Phi, real_t [:,::1] Pi, \
                  real_t [:,::1] Omega, real_t [:,::1] dPi_dr, double [::1] a,\
                  int NR, double init_r, double final_r, int NSIDE, \
                  int n_iter = 30, double ang_epsilon = 1e-2, ray_tracing=True,
                  int max_shooting_trials = 10, bool enable_shear=False):

        self.ray_tracing = ray_tracing

        if ray_tracing is True:
            self.c_geo = \
                new _Geodesic[real_t, idx_t] (&Phi[0, 0], &Pi[0, 0],
                                              &Omega[0, 0], &dPi_dr[0, 0], &a[0],
                                              NR, init_r, final_r, NSIDE, n_iter,
                                              ang_epsilon, max_shooting_trials,
                                              enable_shear)
        else:
            raise ValueError('Now geodesic class ONLY does ray-tracing!')

    def __dealloc__(self):
        if(self.ray_tracing is True):
            del self.c_geo

    def init_with_healpix_tars(self, double r):
        if (self.ray_tracing is True):
            self.c_geo.init_with_healpix_tars(r)
        else:
            print("Doing nothing since in ray-tracing-mode!")
            pass

    def shoot(self):
        if (self.ray_tracing is True):
            self.c_geo.shoot()
        else:
            print("Doing nothing since in ray-tracing-mode!")

    ###############Functions getting class elements####################
    def DA(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p]>self.c_geo.DA_a)
        #npy_arr = npy.asarray(<npy.float64_t[:n_p]> self.c_geo.DA_a)

        return npy_arr;

    def dDAdt(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p]> self.c_geo.dDAdt_a)

        return npy_arr;

    def k0(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p]> self.c_geo.k0_a)

        return npy_arr;

    def z(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p]> self.c_geo.z)

        return npy_arr;

    def ntheta_i(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p*2]> self.c_geo.ang_corrs)

        return npy_arr;

    def nphi_i(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<double[:n_p*2]> self.c_geo.ang_corrs)

        return npy_arr;

    def shooting_states(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<int[:n_p]>self.c_geo.tars_lower_bins)

        return npy_arr;
