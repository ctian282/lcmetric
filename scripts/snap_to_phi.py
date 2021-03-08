import numpy as npy
import scipy as scpy
import pickle as pl
import os.path

from nbodykit.source.catalog import Gadget1Catalog
import nbodykit.algorithms as alg

import lcmetric.utils as ut


def snap_to_phi(path, cosmo_paras, L_snap, N_snap, marker):
    L_unit = 3e5
    hL_unit = L_unit * cosmo_paras['h']
    L_snap /= hL_unit

    f = Gadget1Catalog(path,
                       columndefs=[
                           (
                               'Position',
                               ('auto', 3),
                               'all',
                           ),
                           (
                               'GadgetVelocity',
                               ('auto', 3),
                               'all',
                           ),
                           (
                               'ID',
                               'auto',
                               'all',
                           ),
                       ])
    f['Position'] *= 1 / hL_unit
    f.attrs['BoxSize'] = L_snap
    z = f.attrs['Redshift']
    f.attrs['Nmesh'] = N_snap
    print('BoxSize is ' + str(L_snap) + ', and redshift is ' + str(z))
    a = 1 / (1 + z)
    N_snap_part = f['Position'].shape[0]
    rf = ((f.to_mesh(N_snap).to_real_field(normalize=False) *
           (N_snap**3 / N_snap_part) - 1.0) *
          (1.5 * (cosmo_paras['h'] * 100)**2 * cosmo_paras['Omega_m'] / a))
    Phi = ut.inverse_Lap(rf, L_snap, N_snap)
    snap_pw = alg.fftpower.FFTPower(f, mode='1d')

    dir = os.path.dirname(path)
    with open(dir + '/pw_' + marker + str(int(z * 1000)) + '.dat',
              'wb') as file:
        pl.dump(snap_pw, file)
    file.close()

    with open(dir + '/Phi_' + marker + str(int(z * 1000)) + '.dat',
              'wb') as file:
        pl.dump(Phi, file)
    file.close()
