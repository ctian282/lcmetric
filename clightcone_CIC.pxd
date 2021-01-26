cdef extern from "lightcone_CIC.h":

     ctypedef double real_t
     ctypedef int idx_t

     void CIC_deposit(real_t *particles, real_t *origin, real_t *delta, real_t *vw,
                 idx_t npartiles, real_t count_density,
                 real_t max_r, idx_t NR, idx_t NSIDE)

