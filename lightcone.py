import numpy as npy
import scipy as scpy

from nbodykit.source.catalog import Gadget1Catalog
from nbodykit.source.catalog import CSVCatalog

from healpy import hp

import lcmetric.metric as met
import lcmetric.geodesic as geo


def class Lightcone:

    def __init__(self):
        pass

    # 1/H(z)
    def Hint(z):
        return 1 / (self.cosmo_paras['h']*100 \
                    *npy.sqrt(self.cosmo_paras['Omega_m'] * (1+z)**3
                              + self.cosmo_paras['Omega_L']))

    # H(z)
    def H(z):
        return (self.cosmo_paras['h']*100 / (1+z) \
                * npy.sqrt(self.cosmo_paras['Omega_m'] * (1+z)**3
                           + self.cosmo_paras['Omega_L']))

    # integrate 1/H(z) - r, to inversely solve z for given r
    def inverse_Hint(z, r):
        return npy.abs(scpy.integrate.quad(Hint, 0, z,args=(cosmo_paras))[0] - r)

    def read_snap(self, path, snap_type, a):
        if(snap_type == 'CSV'):
            ## Start reading initial snapshot and setting initial data
            f = CSVCatalog( \
                            path,['x','y','z','vx','vy','vz','phi'])

            f['Position'] = (f['x'][:, None]) * [1/cosmo_paras['h']/L_unit, 0, 0] \
                + (f['y'][:, None]) * [0, 1/cosmo_paras_paras['h']/L_unit, 0] \
                + (f['z'][:, None]) * [0, 0, 1/cosmo_paras['h']/L_unit]

            f.attrs['BoxSize'] = self.L_snap
            rf = (f.to_mesh(self.N_snap).to_real_field(normalize=False) \
                  * ( self.N_snap**3 / f['Position'].shape[0] )  - 1.0 ) \
                * (1.5 * (self.cosmo_paras['h']*100)**2 * self.cosmo_paras['Omega_m'] / a )
        else:
            raise ValueError('Snap data type is not supported!')

    def init_from_cone(self, cone_path, Phi_i_path, Phi_f_path, cosmo_paras, \
                       L_snap, N_snap, N_snap_part, \
                       init_z, final_z, NR, NSIDE,\
                       zel_z = None, Phi_zel_path = None, \
                       cone_type = 'CSV', snap_type = 'Gadget1'):
        self.N_snap = N_snap
        self.N_snap_part = N_snap_part
        self.init_z = init_z
        self.final_z = final_z
        self.zel_z = zel_z
        self.init_a = 1 / (1+self.init_z)
        self.final_a = 1 / (1+self.final_z)


        # Initial comoving distance
        self.init_r = scpy.integrate.quad(self.Hint, 0, init_z, args=(cosmo_paras))[0]

        # Final comoving distance
        self.final_r = scpy.integrate.quad(self.Hint, 0, final_z, args=(cosmo_paras))[0]

        # Initial and final Hubble
        self.Hi = self.H(self.init_z, self.cosmo_paras)
        self.Hf = self.H(self.init_z, self.cosmo_paras)

        


