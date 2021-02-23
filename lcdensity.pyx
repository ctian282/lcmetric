import numpy as npy
import healpy as hp
import numexpr as ne
import numba as nb

from nbodykit.source.catalog import Gadget1Catalog
import lcmetric.utils as ut

cimport numpy as npy

cimport clcdensity

from libcpp cimport bool

cdef class DenFromSnaps:
    cdef n_snaps
    cdef list files

    def __cinit__(self, names):
        self.n_snaps = len(names)
        self.files = list()
        for file in names:
            self.files.append(Gadget1Catalog(file,columndefs=[ ('Position', ('auto', 3), 'all',),
                                               ('GadgetVelocity',  ('auto', 3), 'all', ),
                                               ('ID', 'auto', 'all', ),]) )
        #Check if the input data are in increasing order
        for fi in range(1, self.n_snaps):
            print(self.files[fi].attrs['Redshift'])
            if(self.files[fi].attrs['Redshift'] <= self.files[fi-1].attrs['Redshift']):
                raise ValueError('The redshifts are not increasing for input snaps!')

    def build_lc(self ):
        pass


