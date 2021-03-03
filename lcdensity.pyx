import numpy as npy
import healpy as hp
import scipy as scpy

from nbodykit.source.catalog import Gadget1Catalog
import lcmetric.utils as ut

cimport numpy as npy

cimport clcdensity

from libcpp cimport bool
from libcpp.vector cimport vector


cdef class DensFromSnaps:

    cdef _DensFromSnaps *c_dens
    cdef n_snaps
    cdef list files
    cdef dict cosmo_paras

    cdef real_t L_snap, L_unit
    cdef idx_t N_part

    cdef real_t pre_tau

    def __cinit__(self, files,
                  real_t[:] origin, real_t L_snap, idx_t N_part, real_t init_r, cosmo_paras,
                  real_t L_unit):
        self.cosmo_paras = cosmo_paras
        self.L_snap = L_snap
        self.N_part = N_part
        self.L_unit = L_unit
        self.n_snaps = len(files)
        self.files = files

        cdef real_t[:] box = npy.array([L_snap, L_snap, L_snap])

        self.c_dens = new _DensFromSnaps(N_part, &origin[0], init_r, &box[0])

    def proc_snap(self, int fi, real_t tau, real_t dtau, pre_clear = False):

        if(fi >= len(self.files) or fi < 0): return False

        if(pre_clear is True):
            self.c_dens.clear_lc()

        cdef real_t [:,::1] pos, vel
        cdef idx_t [:] IDs
        #for fi in range(len(self.files) - 1, -1, 0):

        z = self.files[fi].attrs['Redshift']
        a = 1 / (1 + z)
        H = ut.H(z, self.cosmo_paras['h']*100,
                 self.cosmo_paras['Omega_m'], self.cosmo_paras['Omega_L'])
        pos = npy.ascontiguousarray(self.files[fi]['Position'] / self.L_unit, dtype=npy.double)
        vel = npy.ascontiguousarray(self.files[fi]['GadgetVelocity'] * (npy.sqrt(a) / (3e5)),
                                    dtype=npy.double)
        IDs = npy.ascontiguousarray(self.files[fi]['ID'])
        #print(str(z)+' '+str(a)+' '+str(H)+' '+str(tau)+' '+str(dtau))
        self.c_dens.proc_snap(
            &pos[0, 0], &vel[0, 0], &IDs[0],
            tau, dtau, a, H)

        return True

    def clear_lc(self):
        self.c_dens.clear_lc()

    def p_num(self):
        return self.c_dens.lc_p.size()

    def get_pdata(self):
        arr = \
            <real_t [:self.c_dens.lc_p.size()]> self.c_dens.lc_p.data()
        return arr
