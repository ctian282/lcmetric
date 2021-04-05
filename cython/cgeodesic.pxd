from libcpp cimport bool


cdef extern from "c++/geodesic.h":

# Declare the class with cdef
    cdef cppclass _Geodesic[T, IT]:

        _Geodesic(T *Phi_in, T *Pi_in, T *Omega_in, T *dPi_dr_in,
                  double * a_in, int NR_in, double init_r_in, double final_r_in,
                  int NSIDE_in, int n_iter_in, double ang_epsilon_in,
                  int n_max_shooting_in, bool enable_shear_in, bool use_CIC) except +

        void init_with_healpix_tars(double r, int nside) except +
        void shoot() except +



        int NR;
        int NSIDE;
        int NPIX;
        int niter;
        int lmax
        int n_p;
        int n_alm_idx;

        int *tars_lower_bins;


        double init_r, final_r;
        double dr, ang_epsilon;


        double *DA_a;
        double *dDAdt_a;
        double *k0_a;
        double *z;

        double *ang_corrs;
        double *diff_s_ang;
