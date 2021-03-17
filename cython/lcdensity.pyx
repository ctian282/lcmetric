import numpy as npy
import healpy as hp
import scipy as scpy

from nbodykit.source.catalog import Gadget1Catalog
import lcmetric.utils as ut

cimport numpy as npy

cimport clcdensity

from libcpp cimport bool
from libcpp.vector cimport vector

ctypedef long long idx_t
ctypedef double real_t

cdef class DensFromSnaps:

    cdef _DensFromSnaps[real_t, idx_t] *c_dens

    cdef dict cosmo_paras

    cdef double L_snap, L_unit
    cdef idx_t N_part

    cdef bool is_first_snap

    def __cinit__(self,
                  double[:] origin, double L_snap, idx_t N_part, double init_r,
                  cosmo_paras, double L_unit):
        self.cosmo_paras = cosmo_paras
        self.L_snap = L_snap
        self.N_part = N_part
        self.L_unit = L_unit

        cdef double[:] box = npy.array([L_snap, L_snap, L_snap])

        self.c_dens = new _DensFromSnaps[real_t, idx_t](
            N_part, &origin[0], init_r, &box[0])
        self.is_first_snap = True

    def __dealloc__(self):
        del self.c_dens

    def proc_snap_chunk(self, real_t[:, ::1] pos, real_t[:, ::1] vel,
                        idx_t[:] IDs, double tau, double dtau):

        self.c_dens.proc_snap_chunk(
            &pos[0, 0], &vel[0, 0], &IDs[0],
            tau, dtau, 0, 0, pos.shape[0], self.is_first_snap)


    def not_first_snap(self):
        self.is_first_snap = False

    def clear_lc(self):
        self.c_dens.clear_lc()

    def p_num(self):
        return self.c_dens.lc_p.size()

    def get_pdata(self):
        arr = \
            <real_t [:self.c_dens.lc_p.size()]> self.c_dens.lc_p.data()
        return arr
