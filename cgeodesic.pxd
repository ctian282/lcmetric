cdef extern from "geodesic_macros.h":
    pass
cdef extern from "geodesic.h":

    ctypedef double real_t
    ctypedef int idx_t

    
# Declare the class with cdef
    cdef cppclass _Geodesic:
        _Geodesic(real_t *Phi_in, real_t *Pi_in, real_t *Omega_in, real_t *dPi_dr_in,
                  real_t * a_in, idx_t NR_in, real_t init_r_in, real_t final_r_in) except +

        _Geodesic(real_t *Phi_in, real_t *Pi_in, real_t *Omega_in, real_t *dPi_dr_in,
                  real_t * a_in, idx_t NR_in, real_t init_r_in, real_t final_r_in,
                  idx_t NSIDE_in, idx_t n_iter_in, real_t ang_epsilon_in) except +

        void init_with_healpix_tars(real_t r) except +
        void shoot() except +
        
  


        idx_t NR, NSIDE, NPIX, n_iter, lmax;
        real_t init_r, final_r;
        real_t dr, ang_epsilon;
        
        idx_t n_p, n_alm_idx;

        real_t * DA_a;
        real_t *dDAdt_a;
        real_t *k0_a;
        real_t *z;
        
