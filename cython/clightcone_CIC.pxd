cdef extern from "c++/lightcone_CIC.h":

     ctypedef long long idx_t

     void CIC_deposit[T](
         T *particles, double *origin, T *delta,
                 T *vw, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, int vx_is_weight) except +

     void kappa_deposit[T](
         T *particles, double *a, double *origin,
                   T *kappa1, T *kappa2, idx_t nparticles,
                   double max_r, double min_r, int NR, int NSIDE) except +

     void CIC_deposit_with_wgt[T](
         T *particles, double *origin, T *delta, T *vw, T *counts,
         idx_t nparticles, double count_density,
         double max_r, double min_r, int NR, int NSIDE, T *weight) except +
