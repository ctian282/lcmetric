cdef extern from "c++/lightcone_CIC.h":

     ctypedef double real_t
     ctypedef int idx_t

     void CIC_deposit(real_t *particles, real_t *origin, real_t *delta, real_t *vw, real_t *counts,
                 idx_t npartiles, real_t count_density,
                 real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE, idx_t vx_is_weight)

     void kappa_deposit(real_t *particles, real_t *a, real_t *origin,
                        real_t *kappa1, real_t *kappa2, idx_t nparticles,
                        real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE)

     void CIC_deposit_with_wgt(
         real_t *particles, real_t *origin, real_t *delta, real_t *vw, real_t *counts,
         idx_t nparticles, real_t count_density,
         real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE, real_t *weight)