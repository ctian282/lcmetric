# distutils: sources = 
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

import lcmetric.utils as ut

cimport numpy as npy

import lcmetric.cgeodesic as geo
import lcmetric.clightcone_CIC as lc_CIC
import lcmetric.utils as ut
import lcmetric.cutils as utC

from libcpp cimport bool
#from cgeodesic cimport _Geodesic

class Lensing:

    def __init__(self, met, init_r, final_r, NR, NSIDE,
                 mode = 'ray_tracing', **kwargs):
        self.met = met
        self.init_r = init_r
        self.final_r = final_r
        self.NR = NR
        self.NSIDE = NSIDE
        self.NPIX = 12 * self.NSIDE**2
        self.mode = mode

        if(mode == 'ray_tracing'):
            try:
                ang_epsilon = kwargs['ang_epsilon']
            except:
                ang_epsilon = 1e-7

            self.Omega, self.Omega_dot, self.Pi_dot, self.dPi_dr \
                = self.gen_fields_for_rt()

            self.Phi = self.met.sols['Phi']
            self.Pi = self.met.sols['Pi']
            self.a = self.met.metric_f['a']
            self.geo = geo.Geodesic(self.Phi, self.Pi, self.Omega,
                                    self.dPi_dr, self.a, self.NR,
                                    self.init_r, self.final_r, self.NSIDE,
                                    ang_epsilon = 1e-7)
        elif(mode is 'born_approx_lc'):
            self.kappa1 = kwargs['kappa1']
            self.kappa2 = kwargs['kappa2']
            self.L_snap = kwargs['L_snap']
            self.N_snap_part = kwargs['N_snap_part']
        elif(mode is 'born_approx_snap'):
            self.Phi = self.met.sols['Phi']
            self.lmax = 2*NSIDE - 1

            lm = hp.Alm.getlm(self.lmax)
            self.lm = -lm[0] *(lm[0] + 1)

            if('dr_list' in kwargs):
                self.dr_list = kwargs['dr_list']
            else:
                self.dr_list = npy.asarray(npy.zeros(NR+1))
                for n in range(self.NR+1):
                    self.dr_list[n] = (self.init_r - self.final_r) / self.NR
        else:
            raise ValueError('Mode can not be processed!')


##############Doing lensing potential thing########################

    def to_tau(self, ntau):
        """
        From mesh idx to radius
        """
        return self.final_r + ntau / self.NR * (self.init_r - self.final_r)

    def lensing_pot_int(self, r, nr):
        return -(r - self.to_tau(nr)) * (2 * self.Phi[nr]) / (r * self.to_tau(nr))

    def gen_snaps_lensing_conv(self, r):
        if( r < self.final_r or r > self.init_r):
            raise ValueError('r is too large or too small!')

        Psi = npy.zeros(self.Phi.shape[1])

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


    def set_sources(self, **kwargs):
        if(kwargs['init_with_hp_tars'] == True):
            self.r = kwargs['r']

            if( self.r < self.final_r or self.r > self.init_r):
                raise ValueError('r is too large or too small!')

            if(self.mode is 'ray_tracing'):
                self.geo.init_with_healpix_tars(self.r)
            elif(self.mode is 'born_approx_snap'):
                pass
        elif(kwargs['init_with_input_tars'] == True):
            ValueError('The target type has not been supported yet')

    def gen_lc_lensing_conv(self, r):
        resol = 4*npy.pi / (self.NPIX)
        temp = \
            1.5 * self.met.Hubble_0 ** 2 * self.met.Omega_m * \
            self.L_snap**3 / self.N_snap_part / resol * \
            (self.kappa1 - self.kappa2 / r)
        lower_r_bin = \
            int(npy.floor( (r - self.final_r) / ( (self.init_r - self.final_r)/ self.NR) ))
        res = temp[0:lower_r_bin+1,:].sum(axis = 0)
        lw = r - lower_r_bin * ((self.init_r - self.final_r)/ self.NR)
        res += temp[lower_r_bin] * (1 - lw) + temp[lower_r_bin + 1] * lw;
        return res

    def calculate(self):
        if(self.mode is 'ray_tracing'):
            print('Start shooting!')
            self.geo.shoot()
            return self.geo.DA()
        elif(self.mode is 'born_approx_snap'):
            return self.gen_snaps_lensing_conv(self.r)
        elif(self.mode is 'born_approx_lc'):
            return self.gen_lc_lensing_conv(self.r)

    def gen_fields_for_rt(self):
        Omega = ut.np_fderv1(
            self.met.sols['Phi'], -(self.init_r - self.final_r) / self.NR, 0)
        Omega_dot = ut.np_fderv2(
            self.met.sols['Phi'], -(self.init_r - self.final_r) / self.NR, 0)
        Pi_dot = ut.np_fderv1(
            self.met.sols['Pi'], -(self.init_r - self.final_r) / self.NR, 0)
        dPi_dr = -2 * Pi_dot - Omega_dot - 3 * self.met.Hubble_hier[0][:,None] * \
            (Omega + self.met.sols['Pi']) \
            - (2 * self.met.metric_f['Hubble_dt'][:,None] \
               + self.met.metric_f['Hubble'][:,None]**2) * self.met.sols['Phi']
        return (Omega, Omega_dot, Pi_dot, dPi_dr)


