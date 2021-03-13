cdef extern from "c++/utils.h":

     ctypedef double real_t
     ctypedef int idx_t
     void _interp(real_t *d, real_t *dx, idx_t nx, idx_t ny, idx_t nz,
             real_t *x_list, idx_t ns, real_t *res) except +

     void _test_openmp() except +
