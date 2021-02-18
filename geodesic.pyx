# distutils: sources = 
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

import lcmetric.utils as ut

cimport numpy as npy

cimport cgeodesic
#from cgeodesic cimport _Geodesic

cdef class Geodesic:
    cdef _Geodesic *c_geo

    
    
    def __cinit__(self, real_t [:,::1] Phi, real_t [:,::1] Pi, \
                  real_t [:,::1] Omega, real_t [:,::1] dPi_dr, real_t [::1] a,\
                  idx_t NR, real_t init_r, real_t final_r, idx_t NSIDE, \
                  idx_t n_iter = 30, real_t ang_epsilon = 1e-2):
        self.c_geo = \
            new _Geodesic(&Phi[0,0], &Pi[0,0], \
                          &Omega[0,0], &dPi_dr[0,0], &a[0], \
                          NR, init_r, final_r, NSIDE, n_iter, ang_epsilon)

    def __dealloc__(self):
        del self.c_geo

    def init_with_healpix_tars(self, real_t r):
        self.c_geo.init_with_healpix_tars(r)

        
    def shoot(self):
        self.c_geo.shoot()











    ###############Functions getting class elements####################    
    def DA(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<npy.float64_t[:n_p]> self.c_geo.DA_a)

        return npy_arr;

    def dDAdt(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<npy.float64_t[:n_p]> self.c_geo.dDAdt_a)

        return npy_arr;

    def k0(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<npy.float64_t[:n_p]> self.c_geo.k0_a)

        return npy_arr;

    def z(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = npy.asarray(<npy.float64_t[:n_p]> self.c_geo.z)

        return npy_arr;

        
    
        
        
