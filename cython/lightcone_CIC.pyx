# distutils: sources =
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

cimport clightcone_CIC as lc_CIC

import numpy as npy



cpdef deposit(real_t [:,::1] particles, real_t [:] origin, real_t [:,::1] delta, \
              real_t density_count,                                          \
              real_t [:,::1] vw, real_t [:,::1] counts, \
              real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE, idx_t vx_is_weight = 0):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0], &counts[0,0],   \
                       nparticles, density_count,                            \
                       max_r, min_r, NR, NSIDE, vx_is_weight)

cpdef lensing_kappa_deposit(real_t [:,::1] particles, real_t [::1] a, real_t [:] origin,\
              real_t [:,::1] kappa1, real_t [:,::1] kappa2, \
              real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE):
    nparticles = particles.shape[0]
    lc_CIC.kappa_deposit(&particles[0,0], &a[0], &origin[0],
                         &kappa1[0,0], &kappa2[0,0],
                         nparticles,
                         max_r, min_r, NR, NSIDE)

cpdef deposit_with_wgt(real_t [:,::1] particles, real_t [:] origin, real_t [:,::1] delta,
                       real_t density_count, real_t [:,::1] vw, real_t [:,::1] counts,
                       real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE,
                       real_t [:,::1] wgt):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit_with_wgt(&particles[0,0], &origin[0], &delta[0,0],
                                &vw[0,0], &counts[0,0],
                                nparticles, density_count,
                                max_r, min_r, NR, NSIDE, &wgt[0, 0])
