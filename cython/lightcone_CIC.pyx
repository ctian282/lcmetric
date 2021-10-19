# distutils: sources =
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

cimport clightcone_CIC as lc_CIC

import numpy as npy

cimport cython
ctypedef cython.floating real_t


cpdef deposit(real_t [:,::1] particles, double [::1] origin, real_t [:,::1] delta, \
              double density_count,                                          \
              real_t [:,::1] vw, real_t [:,::1] counts, \
              double max_r, double min_r, int NR, int NSIDE, depo_method, double shift=0):

    nparticles = particles.shape[0]
    if depo_method is 'CIC':
        lc_CIC.CIC_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0], &counts[0,0],
                           nparticles, density_count,
                           max_r, min_r, NR, NSIDE, shift)
    elif depo_method is 'NGP':
        lc_CIC.NGP_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0], &counts[0,0],
                           nparticles, density_count,
                           max_r, min_r, NR, NSIDE, shift)
    else:
        raise ValueError('Depo method is not supported')

cpdef deposit_ang(real_t [:,::1] particles, double [::1] origin, real_t [:,::1] delta, \
              double density_count,                                          \
                  real_t [:,::1] vw, real_t [:,::1] vtheta, real_t [:,::1] vphi, real_t [:,::1] counts, \
              double max_r, double min_r, int NR, int NSIDE, depo_method, double shift=0):
        lc_CIC.CIC_deposit_ang(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0], &counts[0,0],
                           nparticles, density_count,
                           max_r, min_r, NR, NSIDE, shift)



cpdef lensing_kappa_deposit(real_t [:,::1] particles, double [::1] a, double [::1] origin,\
              real_t [:,::1] kappa1, real_t [:,::1] kappa2, \
                            double max_r, double min_r, int NR, int NSIDE, double shift=0):
    nparticles = particles.shape[0]
    lc_CIC.kappa_deposit(&particles[0,0], &a[0], &origin[0],
                         &kappa1[0,0], &kappa2[0,0],
                         nparticles,
                         max_r, min_r, NR, NSIDE, shift)

cpdef deposit_with_wgt(real_t [:,::1] particles, double [::1] origin, real_t [:,::1] delta,
                       double density_count, real_t [:,::1] vw, real_t [:,::1] counts,
                       double max_r, double min_r, int NR, int NSIDE,
                       real_t [:,::1] wgt, double shift=0):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit_with_wgt(&particles[0,0], &origin[0], &delta[0,0],
                                &vw[0,0], &counts[0,0],
                                nparticles, density_count,
                                max_r, min_r, NR, NSIDE, &wgt[0, 0], shift)
