import numpy as npy
import scipy as scpy
import pickle as pl

from nbodykit.source.catalog import Gadget1Catalog
from nbodykit.source.catalog import CSVCatalog

import healpy as hp

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
                          depth = 5, n_vcycles = 100, npre = 16, \
                          lmax = 2*NSIDE - 1, verbose = False)
        try:
            if(keyws['from_files'] == True):
                self.Phi_i = read_pickle(keyws['Phi_i_path'])
                self.Pi_i = read_pickle(keyws['Pi_i_path'])
                self.Phi_f = read_pickle(keyws['Phi_f_path'])
                self.delta = read_picles(keyws['delta_path'])
                self.vw = read_picles(keyws['vw_path'])
                self.init_z = keyws['init_z']
                self.final_z = keyws['final_z']

                self.init_r = scpy.integrate.quad(self.Hint, 0, init_z)[0]
                # Final comoving distance
                self.final_r = scpy.integrate.quad(self.Hint, 0, final_z)[0]

                self.met.init_from_slice(
                    self.init_z, self.init_r, self.delta,
                    self.vw, self.Phi_i, self.Pi_i, self.cosmo_paras,
                    self.final_z, self.final_r, self.Phi_f)
        except:
            pass

    def read_pickle(self, path):
        with open(path, 'rb') as f:
            Phi_i = pickle.load(f)
        f.close()
        return Phi_i

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


class LightconeFromCone(Lightcone):


    def __init__(self, cone_path, Phi_i_path, Phi_f_path, origin,
                 cosmo_paras, L_snap, N_snap,
                 init_z, final_z, NR, NSIDE,
                 zel_z=None, Phi_zel_path=None,
                 cone_type='CSV', snap_type='Gadget1'):

        Lightcone.__init__(self, origin, cosmo_paras, L_snap, N_snap, NR, NSIDE)

        self.init_z = init_z
        self.final_z = final_z
        self.zel_z = zel_z
        self.init_a = 1 / (1+self.init_z)
        self.final_a = 1 / (1+self.final_z)

        print(self.L_snap)
        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0, init_z)[0]

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0, final_z)[0]

        if(zel_z is not None):
            self.zel_a = 1 / (1 + zel_z)
            self.zel_r = scpy.integrate.quad(self.Hint, 0, zel_z)[0]

        # Initial and final Hubble
        self.Hi = self.H(self.init_z)
        self.Hf = self.H(self.init_z)

        # print(cone_type)
        print("Starting reading initial snap")
        self.Phi_i, self.Pi_i = self.Phi_Pi_gen(
            self.read_snap_density(Phi_i_path, snap_type, self.init_a),
            self.init_r, self.theta_list, self.phi_list)

        print("Starting reading final snap")
        self.Phi_f, self.Pi_f = self.Phi_Pi_gen(
            self.read_snap_density(Phi_f_path, snap_type, self.final_a),
            self.final_r, self.theta_list, self.phi_list)

        print("Finishing reading snaps")

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
        self.delta, self.vw = \
            self.read_lc_density(cone_path, cone_type, rf_i0, rf_i1, rf_i2)

        self.met.init_from_slice(self.init_z, self.init_r, self.delta,
                                 self.vw, self.Phi_i, self.Pi_i, self.cosmo_paras,
                                 self.final_z, self.final_r, self.Phi_f)




    def read_lc_density(self, path, cone_type,
                        rf_i0=None, rf_i1=None, rf_i2=None):

        if(cone_type == 'CSV'):
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
        else:
            raise ValueError('Cone data type is not supported')

        f.attrs['BoxSize'] = self.L_snap
        f['Particles'] = f['Particles'].rechunk(10000000)
        delta = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        delta.fill(-1)
        vw = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        counts = npy.zeros((self.NR+2, self.NPIX), dtype = npy.double)
        count_density = 1 / (self.L_snap / self.N_snap)**3 \
            / (self.N_snap**3 / self.N_snap_part)

        self.N_lc_part = f['Particles'][0]

        for d in f['Particles'].partitions:
            pdata = d.compute()
            if(self.zel_z is not None):
                pdata += \
                    utC.interp(rf_i0, self.dx,
                               npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                    * [1,0,0,0,0,0] \
                    + utC.interp(rf_i1, self.dx,
                                 npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                    * [0,1,0,0,0,0] \
                    + utC.interp(rf_i2, self.dx,
                                 npy.ascontiguousarray(pdata[:, 0:3]))[:, None] \
                    * [0,0,1,0,0,0]
            lc_CIC.deposit(pdata, self.origin, delta, count_density, \
                           vw, counts, self.init_r, self.final_r, self.NR,
                           self.NSIDE, 0)

        vw /= counts
        vw = npy.nan_to_num(vw)
        return (delta, vw)




