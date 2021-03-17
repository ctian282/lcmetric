# distutils: sources =
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

cimport clightcone_CIC as lc_CIC

import numpy as npy

cimport cython
ctypedef cython.floating real_t


cpdef deposit(real_t [:,::1] particles, double [::1] origin, real_t [:,::1] delta, \
              double density_count,                                          \
              real_t [:,::1] vw, real_t [:,::1] counts, \
              double max_r, double min_r, int NR, int NSIDE, int vx_is_weight):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0], &counts[0,0],   \
                       nparticles, density_count,                            \
                       max_r, min_r, NR, NSIDE, vx_is_weight)

cpdef lensing_kappa_deposit(real_t [:,::1] particles, double [::1] a, double [::1] origin,\
              real_t [:,::1] kappa1, real_t [:,::1] kappa2, \
              double max_r, double min_r, int NR, int NSIDE):
    nparticles = particles.shape[0]
    lc_CIC.kappa_deposit(&particles[0,0], &a[0], &origin[0],
                         &kappa1[0,0], &kappa2[0,0],
                         nparticles,
                         max_r, min_r, NR, NSIDE)

cpdef deposit_with_wgt(real_t [:,::1] particles, double [::1] origin, real_t [:,::1] delta,
                       double density_count, real_t [:,::1] vw, real_t [:,::1] counts,
                       double max_r, double min_r, int NR, int NSIDE,
                       real_t [:,::1] wgt):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit_with_wgt(&particles[0,0], &origin[0], &delta[0,0],
                                &vw[0,0], &counts[0,0],
                                nparticles, density_count,
                                max_r, min_r, NR, NSIDE, &wgt[0, 0])
