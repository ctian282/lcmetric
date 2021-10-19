cdef extern from "c++/lightcone_CIC.h":

     ctypedef long long idx_t

     void CIC_deposit[T](
         T *particles, double *origin, T *delta,
                 T *vw, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift) except +

     void CIC_deposit_ang[T](
         T *particles, double *origin, T *delta,
                 T *vw, T *vtheta, T*vphi, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift) except +

     void NGP_deposit[T](
         T *particles, double *origin, T *delta,
                 T *vw, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift) except +


     void kappa_deposit[T](
         T *particles, double *a, double *origin,
                   T *kappa1, T *kappa2, idx_t nparticles,
                   double max_r, double min_r, int NR, int NSIDE, double shift) except +

     void CIC_deposit_with_wgt[T](
         T *particles, double *origin, T *delta, T *vw, T *counts,
         idx_t nparticles, double count_density,
         double max_r, double min_r, int NR, int NSIDE, T *weight, double shift) except +
