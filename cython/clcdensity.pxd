from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef long long idx_t
ctypedef float real_t


cdef extern from "c++/lcdensity.h":


    cdef cppclass _DensFromSnaps[T]:
        _DensFromSnaps(idx_t n_part_in, double* obs_in, double init_r_in,
                        double *box) except +
        void proc_snap(T* pos, T* vel, idx_t *ids, double tau,
                       double dtau, T a, T H, bool is_first_snap) except +

        void proc_snap_chunk(T * pos, T * vel, idx_t *ids, double tau,
                             double dtau, T a, T H, idx_t np, bool is_first_snap) except +

        void clear_lc() except +

        double init_r;
        vector[T] lc_p;
