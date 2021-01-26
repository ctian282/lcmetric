# distutils: sources = 
# distutils: include_dirs = ./ /usr/include/healpix_cxx/

cimport clightcone_CIC as lc_CIC

import numpy as npy

cpdef deposit(double [:,::1] particles, double [:] origin, double [:,::1] delta, \
              double density_count,                                          \
              double [:,::1] vw, double max_r, int NR, int NSIDE):
    nparticles = particles.shape[0]
    lc_CIC.CIC_deposit(&particles[0,0], &origin[0], &delta[0,0], &vw[0,0],   \
                       nparticles, density_count,                            \
                       max_r, NR, NSIDE)
    
