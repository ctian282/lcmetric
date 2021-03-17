from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "c++/lcdensity.h":


    cdef cppclass _DensFromSnaps[T, IT]:
        _DensFromSnaps(IT n_part_in, double* obs_in, double init_r_in,
                        double *box) except +
        void proc_snap(T* pos, T* vel, IT *ids, double tau,
                       double dtau, T a, T H, bool is_first_snap) except +

        void proc_snap_chunk(T * pos, T * vel, IT *ids, double tau,
                             double dtau, T a, T H, IT np, bool is_first_snap) except +

        void clear_lc() except +

        double init_r;
        vector[T] lc_p;
