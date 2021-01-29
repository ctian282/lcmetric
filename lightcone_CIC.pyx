# distutils: sources = 
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

cimport clightcone_CIC as lc_CIC

import numpy as npy



cpdef deposit(real_t [:,::1] particles, real_t [:] origin, real_t [:,::1] delta, \
              real_t density_count,                                          \
              real_t [:,::1] vw, real_t max_r, real_t min_r, idx_t NR, idx_t NSIDE, idx_t vx_is_weight = 0):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0],   \
                       nparticles, density_count,                            \
                       max_r, min_r, NR, NSIDE, vx_is_weight)
    
