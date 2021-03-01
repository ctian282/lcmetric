import numpy as npy
import scipy as scpy
import pickle as pl

from nbodykit.source.catalog import Gadget1Catalog
from nbodykit.source.catalog import CSVCatalog

import healpy as hp
import glob
import struct

import lcmetric.metric as met
import lcmetric.cgeodesic as geo
import lcmetric.clightcone_CIC as lc_CIC
import lcmetric.utils as ut
import lcmetric.cutils as utC


class Lightcone:

    def __init__(self, origin, cosmo_paras, L_snap, N_snap,
                 NR, NSIDE, **keyws):
        self.L_snap = L_snap
        self.N_snap = N_snap
        self.NR = NR
        self.NSIDE = NSIDE
        self.NPIX = 12 * self.NSIDE**2

        self.cosmo_paras = cosmo_paras
        self.origin = npy.array(origin, dtype = npy.double)

        self.theta_list, self.phi_list = hp.pix2ang(NSIDE, range(12*NSIDE**2))
        self.dx = npy.array([self.L_snap / self.N_snap]*3)

        # L unit
        self.L_unit = 3e5
        self.hL_unit = self.L_unit * cosmo_paras['h']

        self.met = met.Metric(NSIDE, epsilon = 1e-12, grid = 'healpy', alm_iter = 50, \
                              depth = 5, n_vcycles = 100, npre = 16, npost = 16,\
                              lmax = 2*NSIDE - 1, verbose = False)
        try:
            if(keyws['from_files'] == True):
                self.Phi_i = self.read_pickle(keyws['Phi_i_path'])
                self.Pi_i = self.read_pickle(keyws['Pi_i_path'])
                self.Phi_f = self.read_pickle(keyws['Phi_f_path'])
                self.delta = self.read_pickle(keyws['delta_path'])
                self.vw = self.read_pickle(keyws['vw_path'])
                self.init_z = keyws['init_z']
                self.final_z = keyws['final_z']

                self.init_r = scpy.integrate.quad(self.Hint, 0, self.init_z)[0]
                # Final comoving distance
                self.final_r = scpy.integrate.quad(self.Hint, 0, self.final_z)[0]

                self.met.init_from_slice(
                    self.init_z, self.init_r, self.delta,
                    self.vw, self.Phi_i, self.Pi_i, self.cosmo_paras,
                    self.final_r, self.Phi_f)
        except:
            pass

    @staticmethod
    def read_pickle(path):
        with open(path, 'rb') as f:
            temp = pl.load(f)
        f.close()
        return temp

    @staticmethod
    def dump_pickle(path, field):
        with open(path, 'wb') as f:
            pl.dump(field)
        f.close()

    def dump_all(self, path):
        self.dump_pickle(path + 'Phi_i.dat', self.Phi_i)
        self.dump_pickle(path + 'Phi_f.dat', self.Phi_f)
        self.dump_pickle(path + 'Pi_i.dat', self.Pi_i)
        self.dump_pickle(path + 'delta.dat', self.delta)
        self.dump_pickle(path + 'vw_depo.dat', self.vw)

    # 1/H(z)
    def Hint(self, z):
        return 1 / (self.cosmo_paras['h']*100
                    * npy.sqrt(self.cosmo_paras['Omega_m'] * (1+z)**3
                              + self.cosmo_paras['Omega_L']))

    # H(z)
    def H(self, z):
        return (self.cosmo_paras['h']*100 / (1+z)
                * npy.sqrt(self.cosmo_paras['Omega_m'] * (1+z)**3
                           + self.cosmo_paras['Omega_L']))

    # integrate 1/H(z) - r, to inversely solve z for given r
    def inverse_Hint(self, z, r):
        return npy.abs(scpy.integrate.quad(self.Hint, 0, z)[0] - r)

    def read_snap_density(self, path, snap_type, a):
        if (snap_type == 'CSV'):
            ## Start reading initial snapshot and setting initial data
            f = CSVCatalog(path, ['x','y','z','vx','vy','vz','phi'])

            f['Position'] = (f['x'][:, None]) * [1/self.hL_unit, 0, 0] \
                + (f['y'][:, None]) * [0, 1/self.hL_unit, 0] \
                + (f['z'][:, None]) * [0, 0, 1/self.hL_unit]

            f.attrs['BoxSize'] = self.L_snap
        elif (snap_type == 'Gadget1'):
            f = Gadget1Catalog(
                path,
                columndefs=[('Position', ('auto', 3), 'all',),
                            ('GadgetVelocity',  ('auto', 3), 'all', ),
                            ('ID', 'auto', 'all', ),])
            f['Position'] *= 1/self.hL_unit
            f.attrs['BoxSize'] = self.L_snap
        else:
            raise ValueError('Snap data type is not supported!')

        self.N_snap_part = f['Position'].shape[0]
        return ((f.to_mesh(self.N_snap).to_real_field(normalize=False)
                 * (self.N_snap**3 / self.N_snap_part) - 1.0)
                * (1.5 * (self.cosmo_paras['h']*100)**2
                   * self.cosmo_paras['Omega_m'] / a))

    def Phi_Pi_gen(self, rf, r, theta_list, phi_list):
        x_list = npy.array([ \
                             r * npy.sin(theta_list) * npy.cos(phi_list) + self.origin[0],        \
                             r * npy.sin(theta_list) * npy.sin(phi_list) + self.origin[1],        \
                             r * npy.cos(theta_list) + self.origin[2]                             \
                            ])
        x_list = npy.ascontiguousarray(npy.transpose(x_list))

        Phi = ut.inverse_Lap(rf, self.L_snap, self.N_snap)
        Pi = ut.f_r_derv(Phi, npy.array([self.L_snap / self.N_snap] * 3), \
                                r, x_list)
        Phi = ut.interp(Phi, self.dx, x_list)

        return (Phi, Pi)

    def build_lcmetric(self):
        self.met.build_lcmetric()

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

    def lensing(self, mode = 'ray_tracing', **kwargs):
        if(mode == 'ray_tracing'):
            if(kwargs['init_with_hp_tars'] == True):
                r = kwargs['r']
                try:
                    ang_epsilon = kwargs['ang_epsilon']
                except:
                    ang_epsilon = 1e-7
                Omega, Omega_dot, Pi_dot, dPi_dr = self.gen_fields_for_rt()

                Phi = self.met.sols['Phi']
                Pi = self.met.sols['Pi']
                a = self.met.metric_f['a']
                self.geo = geo.Geodesic(Phi, Pi, Omega, dPi_dr, a, self.NR,
                                        self.init_r, self.final_r, self.NSIDE,
                                        ang_epsilon = 1e-7)
                self.geo.init_with_healpix_tars(r)
                print('Start shooting!')
                self.geo.shoot()
            elif(kwargs['init_with_input_tars'] == True):
                ValueError('The target type has not been supported yet')
        elif(mode is 'born_approx_snap'):
            if(kwargs['']):
                pass
        elif(mode is 'born_approx_lc'):
            if(kwargs['']):
                pass
        else:
            raise ValueError('Mode can not be processed!')

class LightconeFromCone(Lightcone):

    def to_r(self, n, nr):
        return n / nr * (self.init_r - self.final_r) + self.final_r

    def __init__(self, cone_path, Phi_i_path, Phi_f_path, origin,
                 cosmo_paras, L_snap, N_snap,
                 init_z, final_z, NR, NSIDE,
                 zel_z=None, Phi_zel_path=None,
                 cone_type='CSV', snap_type='Gadget1', lensing_kappa=False):

        Lightcone.__init__(self, origin, cosmo_paras, L_snap, N_snap, NR, NSIDE)

        self.init_z = init_z
        self.final_z = final_z
        self.zel_z = zel_z
        self.init_a = 1 / (1+self.init_z)
        self.final_a = 1 / (1+self.final_z)

        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0, init_z)[0]

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0, final_z)[0]

        self.a=npy.array([1/(1+scpy.optimize.minimize_scalar(
            self.inverse_Hint, args=(self.to_r(n, self.NR))).x) \
                          for n in range(NR+1) ])

        if(zel_z is not None):
            self.zel_a = 1 / (1 + zel_z)
            self.zel_r = scpy.integrate.quad(self.Hint, 0, zel_z)[0]

        # Initial and final Hubble
        self.Hi = self.H(self.init_z)
        self.Hf = self.H(self.init_z)

        print("Starting reading initial snap")
        self.Phi_i, self.Pi_i = self.Phi_Pi_gen(
            self.read_snap_density(Phi_i_path, snap_type, self.init_a),
            self.init_r, self.theta_list, self.phi_list)

        print("Starting reading final snap")
        self.Phi_f, self.Pi_f = self.Phi_Pi_gen(
            self.read_snap_density(Phi_f_path, snap_type, self.final_a),
            self.final_r, self.theta_list, self.phi_list)

        # print("Finishing reading snaps")
        print("Starting reading lightcone data")

        rf_i0 = None
        rf_i1 = None
        rf_i2 = None

        if(zel_z is not None):
            Phi_snap_I = \
                self.read_snap_density(Phi_zel_path, snap_type, self.zel_a)

            Phi_snap_I = 5*ut.inverse_Lap(Phi_snap_I, self.L_snap, self.N_snap)

            rf_i0 = npy.ascontiguousarray(ut.inverse_derv(Phi_snap_I, L_snap, N_snap, 0))
            rf_i1 = npy.ascontiguousarray(ut.inverse_derv(Phi_snap_I, L_snap, N_snap, 1))
            rf_i2 = npy.ascontiguousarray(ut.inverse_derv(Phi_snap_I, L_snap, N_snap, 2))

        # reading light-cone
        self.delta, self.vw, self.counts= \
            self.read_lc_density(cone_path, cone_type, rf_i0, rf_i1, rf_i2,
                                 lensing_kappa=lensing_kappa)

        self.met.init_from_slice(self.init_z, self.init_r, self.delta,
                                 self.vw, self.Phi_i, self.Pi_i, self.cosmo_paras,
                                 self.final_r, self.Phi_f)

    def init_metric():
        self.met.init_from_slice(self.init_z, self.init_r, self.delta,
                                 self.vw, self.Phi_i, self.Pi_i, self.cosmo_paras,
                                 self.final_r, self.Phi_f)


    # Reading lp-cola unformatted files
    def unf_read_file(self, file, p_list=[], np = 7 ):
        with open(file, mode="rb") as f:
            tot_n = 0
            cnt = 0
            while(True):
                cnt += 1
                r=f.read(4)
                if not r: break

                a1 = struct.unpack('i',r)

                r = f.read(a1[0])
                n = struct.unpack('i',r)

                r =f.read(8)
                a,b=struct.unpack('2i',r)

                r=f.read(b)
                p_list.extend(struct.unpack(str(n[0]*np)+'f',r))

                r=f.read(4)
                tot_n += n[0]
        f.close()
        return tot_n

    def read_lc_density(
            self, path, cone_type,
            rf_i0=None, rf_i1=None, rf_i2=None, chunk=20000000, np = 7,
            lensing_kappa=False):
        delta = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        delta.fill(-1)
        vw = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        counts = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        count_density = 1 / (self.L_snap / self.N_snap)**3 \
            / (self.N_snap**3 / self.N_snap_part)

        if(lensing_kappa is True):
            self.kappa1 = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
            self.kappa2 = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)

        if(cone_type == 'UNFORMATTED'):
            files = glob.glob(path)
            self.N_lc_part = 0
            cnt = 0
            pdata = []
            print("file num is "+str(len(files)))
            for i in range(len(files)):
                file = files[i]
                cnt += self.unf_read_file(file, pdata, np)
                if(cnt >= chunk or i == len(files) - 1):
                    pdata = npy.ascontiguousarray(
                        npy.array(pdata).reshape(cnt, np)[:,0:6]) \
                        * npy.array([1/self.hL_unit, 1/self.hL_unit, 1/self.hL_unit,\
                                     1/3e5, 1/3e5, 1/3e5])

                    if(self.zel_z is not None):
                        pdata += \
                            utC.interp(rf_i0, self.dx,\
                                       npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                       * [1,0,0,0,0,0] + \
                            utC.interp(rf_i1, self.dx,
                                         npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                         * [0,1,0,0,0,0] + \
                            utC.interp(rf_i2, self.dx,
                                         npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                         * [0,0,1,0,0,0]
                    lc_CIC.deposit(pdata, self.origin, delta, count_density, \
                                   vw, counts, self.init_r, self.final_r, self.NR,
                                   self.NSIDE, 0)
                    if(lensing_kappa == True):
                        lc_CIC.lensing_kappa_deposit(
                            pdata, self.a, self.origin, self.kappa1, self.kappa2,
                            self.init_r, self.final_r, self.NR, self.NSIDE)
                    pdata = []
                    self.N_lc_part += cnt
                    cnt = 0
            vw /= counts
            vw = npy.nan_to_num(vw)
        elif(cone_type == 'CSV'):
            f = CSVCatalog(path, ['x','y','z','vx','vy','vz','phi'], dtype='f8')
            f['Particles'] = \
                (f['x'][:, None]) \
                * npy.array([1 / self.hL_unit, 0, 0, 0, 0, 0]) \
                + (f['y'][:, None]) \
                * npy.array([0, 1 / self.hL_unit, 0, 0, 0, 0],) \
                + (f['z'][:, None]) \
                * npy.array([0, 0, 1 / self.hL_unit, 0, 0, 0],)\
                + (f['vx'][:, None]) * npy.array([0, 0, 0, 1/3e5, 0, 0],) \
                + (f['vy'][:, None]) * npy.array([0, 0, 0, 0, 1/3e5, 0],) \
                + (f['vz'][:, None]) * npy.array([0, 0, 0, 0, 0, 1/3e5],)

            f.attrs['BoxSize'] = self.L_snap
            f['Particles'] = f['Particles'].rechunk(chunk)

            self.N_lc_part = f['Particles'].shape[0]

            for d in f['Particles'].partitions:
                pdata = d.compute()
                if(self.zel_z is not None):
                    pdata += \
                        utC.interp(rf_i0, self.dx,\
                                   npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                   * [1,0,0,0,0,0] +\
                        utC.interp(rf_i1, self.dx,
                                     npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                     * [0,1,0,0,0,0] +\
                        utC.interp(rf_i2, self.dx,
                                     npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                                     * [0,0,1,0,0,0]
                    lc_CIC.deposit(pdata, self.origin, delta, count_density, \
                                   vw, counts, self.init_r, self.final_r, self.NR,
                                   self.NSIDE, 0)
                    if(lensing_kappa == True):
                        lc_CIC.lensing_kappa_deposit(
                            pdata, self.a, self.origin, self.kappa1, self,kappa2,
                            self.init_r, self.final_r, self.NR, self.NSIDE)
            vw /= counts
            vw = npy.nan_to_num(vw)
        else:
            raise ValueError('Cone data type is not supported')

        return (delta, vw, counts)




