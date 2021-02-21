# distutils: sources = 
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

import lcmetric.utils as ut

cimport numpy as npy

cimport cgeodesic
from libcpp cimport bool
#from cgeodesic cimport _Geodesic

cdef class Geodesic:
    cdef _Geodesic *c_geo

    cdef bool ray_tracing

    cdef npy.ndarray Phi
    cdef npy.ndarray lm

    cdef npy.ndarray dr_list

    cdef real_t init_r, final_r

    cdef idx_t NR, NSIDE, lmax


    def __cinit__(self, real_t [:,::1] Phi, real_t [:,::1] Pi, \
                  real_t [:,::1] Omega, real_t [:,::1] dPi_dr, real_t [::1] a,\
                  idx_t NR, real_t init_r, real_t final_r, idx_t NSIDE, \
                  idx_t n_iter = 30, real_t ang_epsilon = 1e-2, ray_tracing = True,
                  dr_list = None):

        self.ray_tracing = ray_tracing

        if ray_tracing is True:
            self.c_geo = \
                new _Geodesic(&Phi[0,0], &Pi[0,0], \
                              &Omega[0,0], &dPi_dr[0,0], &a[0], \
                              NR, init_r, final_r, NSIDE, n_iter, ang_epsilon)
        else:
            self.Phi = npy.asarray(Phi)
            self.init_r = init_r
            self.final_r = final_r
            self.NR = NR
            self.NSIDE = NSIDE

            self.lmax = 2*NSIDE - 1

            lm = hp.Alm.getlm( \
                    self.lmax)

            self.lm = -lm[0] *(lm[0] + 1)
            if(dr_list == None):
                self.dr_list = npy.asarray(npy.zeros(NR+1))
                for n in range(self.NR+1):
                    self.dr_list[n] = (self.init_r - self.final_r) / self.NR
            else:
                self.dr_list = npy.asarray(dr_list)


    def __dealloc__(self):
        if(self.ray_tracing == True):
            del self.c_geo

    def init_with_healpix_tars(self, real_t r):
        if(self.ray_tracing == True):
            self.c_geo.init_with_healpix_tars(r)
        else:
            print("Doing nothing since in ray-tracing-mode!")
            pass


    def shoot(self):
        if(self.ray_tracing == True):
            self.c_geo.shoot()
        else:
            print("Doing nothing since in ray-tracing-mode!")


##############Doing lensing potential thing########################

    def to_tau(self, ntau):
        """
        From mesh idx to radius
        """
        return self.final_r + ntau / self.NR * (self.init_r - self.final_r)

    def lensing_pot_int(self, r, nr):
        return -(r - self.to_tau(nr)) * (2 * self.Phi[nr]) / (r * self.to_tau(nr))

    def gen_lensing_conv(self, r):
        if( r < self.final_r or r > self.init_r):
            raise ValueError('r is too large or too small!')

        cdef npy.ndarray[real_t, ndim=1, mode='c'] Psi = npy.zeros(self.Phi.shape[1])

        lr_idx = int(npy.floor( (r - self.final_r) / ( (self.init_r - self.final_r)/ self.NR) ))


        for nr in range(lr_idx):
            Psi += self.dr_list[nr] * 0.5 * \
                (self.lensing_pot_int(r, nr) \
                 + self.lensing_pot_int(r, nr+1))

        # Last step
        Psi += (r - self.to_tau(lr_idx)) * self.lensing_pot_int(r, lr_idx) \

        Psi = -0.5 * hp.sphtfunc.alm2map(\
                        (hp.sphtfunc.map2alm(Psi, lmax = self.lmax, iter = 30) \
                         * self.lm ), nside = self.NSIDE)
        return Psi



######################Finishing lensing potential thing##########################






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

    def ntheta_i(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = <npy.float64_t[:n_p*2]> self.c_geo.ang_corrs

        return npy_arr;

    def nphi_i(self):
        cdef int n_p = self.c_geo.n_p;
        npy_arr = <npy.float64_t[:n_p*2]> self.c_geo.ang_corrs

        return npy_arr;


    def shooting_states(self):
        cdef int n_p = self.c_geo.n_p;
        #cdef npy.ndarray[idx_t, ndim=1, mode='c'] npy_arr
        npy_arr = <idx_t[:n_p]>self.c_geo.tars_lower_bins

        return npy_arr;
