from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "c++/lcdensity.h":

    ctypedef double real_t
    ctypedef long long idx_t

    cdef cppclass _DensFromSnaps:
        _DensFromSnaps(idx_t n_part_in, real_t * obs_in, real_t init_r_in,
                        real_t *box) except +
        void proc_snap(real_t * pos, real_t * vel, idx_t *ids, real_t tau,
                        real_t dtau, real_t a, real_t H) except +

        void clear_lc() except +



        real_t init_r;
        vector[real_t] lc_p;
