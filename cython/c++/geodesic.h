#ifndef GEODESIC_H
#define GEODESIC_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <chrono>
#include "healpix_base.h"
#include "alm_healpix_tools.h"
#include "healpix_map.h"
#include "alm.h"

#include "geodesic_macros.h"

using namespace std::complex_literals;
using namespace std::chrono;



typedef std::complex<double> cmpx_t;


struct Photon{
    // particle ID
    int pid;
    // pointing direction: theta & phi
    pointing pt;
    // current r position and scale factor a
    double r, a;

    // derived fields
    double Phi, dPhi_dr, dPhi_dtheta, dPhi_dphi,
        dPhi_ddtheta, dPhi_ddphi, dPhi_dthetadphi,
        dPhi_drdtheta, dPhi_drdphi,
        Omega, dPhi_ddr, Phi_lap;

    // fields of the particle
    GEODESIC_APPLY_TO_FIELDS(DECLARE_REAL_T);
    GEODESIC_APPLY_TO_COMPLEX_FIELDS(DECLARE_COMPLEX_T);
};


template<class T, class IT> class _Geodesic
{
public:

    int NR, NSIDE, NPIX, n_iter, lmax, n_max_shooting;;
    double init_r, final_r;
    double dr, ang_epsilon;

    // No. of particles
    int n_p;

    // Defining geodesic fields
    GEODESIC_APPLY_TO_FIELDS(RK2_FIELDS_ALL_CREATE);
    GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_COMPLEX_FIELDS_ALL_CREATE);

    // photon redshift
    double *z;

    // Inpu fields
    T *Phi, *Pi, *Omega, *dPi_dr;

    // stores targets' positions and scale factor
    double *tars, *a;
    // initial angular corrections
    double *ang_corrs, *d_diff_ang, *d_diff_tar_ang, *diff_s_ang;
    double max_tar_r;
    // lower radial bin for tars, is also used
    // as indicator to mark if a photon cannot be traced anymore,
    // usually this happens when photon is very close to the pole
    int *tars_lower_bins;

    // angle (theta and phi) of each Healpix pixel
    pointing *ang_list;

    // Healpix map, used to generate alm
    Healpix_Map <double> Phi_s[2], Pi_s[2], Omega_s[2],
        Phi_s_ddr[2], Phi_s_dtheta[2], Phi_s_dphi[2],
        Phi_s_ddtheta[2], Phi_s_ddphi[2], Phi_s_dthetadphi[2],
        Phi_s_drdtheta[2], Phi_s_drdphi[2],
        Phi_s_ang_lap[2], temp_map[2];
    Alm< xcomplex< double > > temp_lm;


    T_Healpix_Base<int> *hp;
    Alm_Base * lm_base;
    int *l_list, *m_list;

    // error logger
    std::ofstream cout;

    // timers
    double hp_time_con, adv_time_con, tot_time_con;

    // if consider shear terms
    bool enable_shear;


    inline IT IDX(int nr, int pix, int NPIX) {
        return (IT)nr*NPIX + (IT)pix;
    }

    inline double to_r(int nr){
        return final_r + (init_r - final_r) * (double)nr / NR;
    }

    _Geodesic(T *Phi_in, T *Pi_in, T *Omega_in, T *dPi_dr_in,
              double * a_in, int NR_in,
              double init_r_in, double final_r_in):cout("debug.txt") {
        return;
    }

    _Geodesic(T *Phi_in, T *Pi_in, T *Omega_in, T *dPi_dr_in,
              double * a_in, int NR_in, double init_r_in, double final_r_in,
              int NSIDE_in, int n_iter_in= 30, double ang_epsilon_in = 1e-6,
              int n_max_shooting_in = 10, bool enable_shear_in = false):
        NR(NR_in),
        NSIDE(NSIDE_in),
        NPIX(12*NSIDE*NSIDE),
        n_iter(n_iter_in),
        lmax(2*NSIDE-1),
        n_max_shooting(n_max_shooting_in),
        init_r(init_r_in),
        final_r(final_r_in),
        dr((init_r - final_r)/(double)NR),
        ang_epsilon(ang_epsilon_in),
        Phi(Phi_in),
        Pi(Pi_in),
        Omega(Omega_in),
        dPi_dr(dPi_dr_in),
        a(a_in),
        temp_lm(lmax, lmax),
        cout("debug.txt"),
        hp_time_con(0),
        adv_time_con(0),
        tot_time_con(0),
        enable_shear(enable_shear_in) {

        for(int i = 0; i < 2; i++)
        {
            Phi_s[i].SetNside(NSIDE, RING);
            Pi_s[i].SetNside(NSIDE, RING);
            Omega_s[i].SetNside(NSIDE, RING);
            temp_map[i].SetNside(NSIDE, RING);
            Phi_s_ddr[i].SetNside(NSIDE, RING);
            Phi_s_dtheta[i].SetNside(NSIDE, RING);
            Phi_s_dphi[i].SetNside(NSIDE, RING);
            Phi_s_ddtheta[i].SetNside(NSIDE, RING);
            Phi_s_ddphi[i].SetNside(NSIDE, RING);
            Phi_s_dthetadphi[i].SetNside(NSIDE, RING);
            Phi_s_drdtheta[i].SetNside(NSIDE, RING);
            Phi_s_drdphi[i].SetNside(NSIDE, RING);
            Phi_s_ang_lap[i].SetNside(NSIDE, RING);
        }

        // Healpix instance
        hp = new T_Healpix_Base<int>
                ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

        ang_list = new pointing[NPIX];
        for(int i = 0; i < NPIX; i++)
            ang_list[i] = hp->pix2ang(i);
    }

    ~_Geodesic() {
        GEODESIC_APPLY_TO_FIELDS(RK2_FIELDS_ALL_DEL);
        GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_FIELDS_ALL_DEL);
        delete hp;
        delete [] ang_list;
        delete [] tars;
        delete [] tars_lower_bins;
        delete [] ang_corrs;
        delete [] d_diff_ang;
        delete [] d_diff_tar_ang;
        delete [] diff_s_ang;
        delete [] z;
    }

    void gen_tars_lower_bins() {
        for(int i = 0; i < n_p; i++)
            tars_lower_bins[i] = (int)( (tars[6*i] - final_r) / dr );
    }

    void alloc_par_arrs() {
        tars = new double [6*n_p]();
        tars_lower_bins = new int [n_p]();
        ang_corrs = new double [2*n_p]();
        d_diff_ang = new double [2*n_p]();
        d_diff_tar_ang = new double [2*n_p]();
        diff_s_ang = new double [n_p]();

        z = new double [n_p]();

        GEODESIC_APPLY_TO_FIELDS_ARGS(RK2_FIELDS_ALL_INIT, n_p);
        GEODESIC_APPLY_TO_COMPLEX_FIELDS_ARGS(RK2_COMPLEX_FIELDS_ALL_INIT, n_p);
    }

    // Targets are healpix pixels at distance r
    void init_with_healpix_tars(double r) {
        max_tar_r = r;
        if(r > init_r)
        {
            std::cout<<"r is too large!"<<std::endl;
            throw(-1);
        }
        n_p = NPIX;
        alloc_par_arrs();
        for(int i = 0; i < NPIX; i++)
        {
            pointing ang = hp->pix2ang(i);

            tars[6*i + 0] = r;
            tars[6*i + 1] = ang.theta;
            tars[6*i + 2] = ang.phi;
            tars[6*i + 3] = 0;
            tars[6*i + 4] = 0;
            tars[6*i + 5] = 0;
        }

        gen_tars_lower_bins();
    }


    // 3 things need to be done by this function
    // 1). Extract the healpix array on this shell
    // 2). Convert the alm coeffs of Phi field on this shell
    // 3). Calculating derv1 of Phi field by healpix
    void update_shell_fields(int n, int c) {
        auto start = system_clock::now();
        for(IT i = IDX(n, 0, NPIX); i < IDX(n, 0, NPIX) + NPIX; i++)
        {
            Phi_s[c][i - IDX(n, 0, NPIX)] = Phi[i];
            Pi_s[c][i - IDX(n, 0, NPIX)] = Pi[i];
            Omega_s[c][i - IDX(n, 0, NPIX)] = Omega[i];
            Phi_s_ddr[c][i - IDX(n, 0, NPIX)] = dPi_dr[i];
        }
        // Calculating first derivatives
        map2alm_iter(Phi_s[c], temp_lm, n_iter);
        alm2map_der1(temp_lm, temp_map[c], Phi_s_dtheta[c], Phi_s_dphi[c]);
        for(int i = 0; i < NPIX; i++)
            Phi_s_dphi[c][i] *= sin(ang_list[i].theta);

        // Calculating lap
        for(int l = 0; l <= lmax; l++)
            for(int m = 0; m <= l; m++)
            {
                //temp_lm[i] *= -l_list[i] * (l_list[i]+1);
                temp_lm(l, m) *= -(double)l *(l+1);
            }
        alm2map(temp_lm, Phi_s_ang_lap[c]);

        // Calculating second and mixing derivatives
        map2alm_iter(Phi_s_dtheta[c], temp_lm, n_iter);
        alm2map_der1(temp_lm, temp_map[c], Phi_s_ddtheta[c], Phi_s_dthetadphi[c]);

        map2alm_iter(Pi_s[c], temp_lm, n_iter);
        alm2map_der1(temp_lm, temp_map[c], Phi_s_drdtheta[c], Phi_s_drdphi[c]);
        for(int i = 0; i < NPIX; i++)
            Phi_s_drdphi[c][i] *= sin(ang_list[i].theta);


        // Calculating second and mixing derivatives
        map2alm_iter(Phi_s_dphi[c], temp_lm, n_iter);
        alm2map_der1(temp_lm, temp_map[c], Phi_s_dthetadphi[c], Phi_s_ddphi[c]);
        for(int i = 0; i < NPIX; i++)
            Phi_s_ddphi[c][i] *= sin(ang_list[i].theta);

        auto end   = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        hp_time_con += (double)(duration.count())
            * microseconds::period::num / microseconds::period::den;
    }


    /**********************RHS of geodesics***************/

    double dtheta_dt(Photon &p) {
        return (1.0 + 2.0 * p.Phi ) * p.ntheta;
    }

    double dphi_dt(Photon &p) {
        return (1.0 + 2.0 * p.Phi ) * p.nphi;
    }

    double dk0_dt(Photon &p) {
        return -4.0 * (p.dPhi_dr + p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
            - 2.0 * (p.Omega + p.dPhi_dr);
    }

    double dnr_dt(Photon &p) {
        return 0.0;
    }

    double dntheta_dt(Photon &p) {
        return 2.0 * p.ntheta * ( 1.0 * p.dPhi_dr+ p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
            - 2.0 * (p.dPhi_dtheta / PW2(p.r)  );
    }

    double dnphi_dt(Photon &p) {
        return 2.0 * p.nphi * ( 1.0 * p.dPhi_dr+ p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
            - 2.0 * (p.dPhi_dphi / PW2(p.r * sin(p.pt.theta)));

    }

    double dDA_dt(Photon &p) {
        return p.dDAdt;
    }

    double ddDAdt_dt(Photon &p) {
        return - dk0_dt(p) * p.dDAdt
            + 1.0 * (1 + 2 * p.Phi) *
            (p.nr * p.nr * p.dPhi_ddr + 2.0 * p.nr * p.ntheta * p.dPhi_drdtheta + 2.0 * p.nr * p.nphi * p.dPhi_drdphi
             + p.ntheta * p.ntheta * p.dPhi_ddtheta + 2.0 * p.ntheta * p.nphi * p.dPhi_dthetadphi
             + p.nphi * p.nphi * p.dPhi_ddphi
             - p.Phi_lap ) * p.DA ;
    }


    double de1r_dt(Photon &p) {
        return p.nr * 2.0 *
            (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
    }
    double de1theta_dt(Photon &p) {
        return p.ntheta * 2.0 *
            (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
    }
    double de1phi_dt(Photon &p) {
        return p.nphi * 2.0 *
            (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
    }

    double de2r_dt(Photon &p) {
        return 1.0 * 2.0 *
            (p.e2r * p.dPhi_dr + p.e2theta * p.dPhi_dtheta + p.e2phi * p.dPhi_dphi);
    }
    double de2theta_dt(Photon &p) {
        return p.ntheta * 2.0 *
            (p.e2r * p.dPhi_dr + p.e2theta * p.dPhi_dtheta + p.e2phi * p.dPhi_dphi);
    }
    double de2phi_dt(Photon &p) {
        return p.nphi * 2.0 *
            (p.e2r * p.dPhi_dr + p.e2theta * p.dPhi_dtheta + p.e2phi * p.dPhi_dphi);
    }


    double dbeta1_dt(Photon &p) {
        return 2.0 * p.beta1 * (p.Omega + p.dPhi_dr)
            - (1 + 2 * p.Phi) * p.a *
            (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
    }

    double dbeta2_dt(Photon &p) {
        return 2.0 * p.beta2 * (p.Omega + p.dPhi_dr)
            - (1 + 2 * p.Phi) * p.a *
            (p.e2r * p.dPhi_dr + p.e2theta * p.dPhi_dtheta + p.e2phi * p.dPhi_dphi);
    }

    cmpx_t dsigma_dt(Photon &p) {
        return 0.0;
        return - dk0_dt(p) * p.sigma + 0.5 * PW2(p.DA) * (1.0 + 2.0 * p.Phi)
            * ((p.e1r * p.e1r - p.e2r * p.e2r + 1i * p.e1r * p.e2r + 1i * p.e2r * p.e1r )
               * p.dPhi_ddr
               + (p.e1r * p.e1theta - p.e2r * p.e2theta + 1i * p.e1r * p.e2theta + 1i * p.e2r * p.e1theta )
               * 2.0 * p.dPhi_drdtheta
               + (p.e1r * p.e1phi - p.e2r * p.e2phi + 1i * p.e1r * p.e2phi + 1i * p.e2r * p.e1phi )
               * 2.0 * p.dPhi_drdphi
               + (p.e1theta * p.e1theta - p.e2theta * p.e2theta + 1i * p.e1theta * p.e2theta + 1i * p.e2theta * p.e1theta )
               * p.dPhi_ddtheta
               + (p.e1theta * p.e1phi - p.e2theta * p.e2phi + 1i * p.e1theta * p.e2phi + 1i * p.e2theta * p.e1phi )
               * 2.0 * p.dPhi_dthetadphi
               + (p.e1phi * p.e1phi - p.e2phi * p.e2phi + 1i * p.e1phi * p.e2phi + 1i * p.e2phi * p.e1phi )
               *  p.dPhi_dthetadphi
                );
    }

    cmpx_t depsilon_dt(Photon &p) {
        return 0.0;
        return 2.0 * p.sigma * sqrt(PW2(std::abs(p.epsilon)) + 4.0) / PW2(p.DA);
    }

    cmpx_t domega_dt(Photon &p) {
        return 0.0;
        return (1i * std::conj(p.epsilon) * p.sigma - p.epsilon * std::conj(p.sigma))/
            (PW2(p.DA) * (4 + 2 * sqrt(4 + PW2(std::abs(p.epsilon)) )));
    }


    /****************************************************/

    void set_photon_values(Photon &p, int p_id, int n, int c) {
        p.pid = p_id;

        GEODESIC_APPLY_TO_FIELDS(SET_LOCAL_VALUES);
        GEODESIC_APPLY_TO_COMPLEX_FIELDS(SET_LOCAL_VALUES);


        p.r = to_r(n);
        p.a = a[n];


        if(p.theta < 0) p.theta = -p.theta;
        else if(p.theta > M_PI) p.theta = 2*M_PI - p.theta;
        p.pt = pointing(p.theta, p.phi);


        p.Phi = Phi_s[c].interpolated_value(p.pt);
        p.dPhi_dr = Pi_s[c].interpolated_value(p.pt);
        p.dPhi_dtheta = Phi_s_dtheta[c].interpolated_value(p.pt);
        p.dPhi_dphi = Phi_s_dphi[c].interpolated_value(p.pt);
        p.Omega = Omega_s[c].interpolated_value(p.pt);

        // Ordinary derivatives
        p.dPhi_ddr = Phi_s_ddr[c].interpolated_value(p.pt);
        p.dPhi_drdtheta = Phi_s_drdtheta[c].interpolated_value(p.pt);
        p.dPhi_drdphi = Phi_s_drdphi[c].interpolated_value(p.pt);
        p.dPhi_ddphi = Phi_s_ddphi[c].interpolated_value(p.pt);
        p.dPhi_ddtheta = Phi_s_ddtheta[c].interpolated_value(p.pt);
        p.dPhi_dthetadphi = Phi_s_dthetadphi[c].interpolated_value(p.pt);

        p.Phi_lap = Phi_s_ang_lap[c].interpolated_value(p.pt) / PW2(p.r)
                + (2 * p.dPhi_dr + p.r * p.dPhi_ddr) / p.r ;
        // Adding connection terms to from covariant derivatives
        p.dPhi_ddr += 0;
        p.dPhi_drdtheta -= p.dPhi_dtheta / p.r;
        p.dPhi_drdphi -= p.dPhi_dphi / p.r;
        p.dPhi_ddtheta -= -p.r * p.dPhi_dr;
        p.dPhi_dthetadphi -= p.dPhi_dphi / tan(p.pt.theta);
        p.dPhi_ddphi -= -p.r * PW2(sin(p.pt.theta)) * p.dPhi_dr - cos(p.pt.theta) * sin(p.pt.theta) * p.dPhi_dtheta;

    }

    void set_inter_slice_photon_values(Photon &p,
                                       int p_id,
                                       int n, int c, double weight_l) {
        p.pid = p_id;
        GEODESIC_APPLY_TO_FIELDS(SET_LOCAL_VALUES);
        GEODESIC_APPLY_TO_COMPLEX_FIELDS(SET_LOCAL_VALUES);

        if(p.theta < 0) p.theta = -p.theta;
        else if(p.theta > M_PI) p.theta = 2*M_PI - p.theta;

        p.r = to_r(n) + weight_l * dr;
        p.a = (1.0 - weight_l) * a[n] + weight_l * a[n+1];

        p.pt = pointing(p.theta, p.phi);
        p.Phi = (1.0 - weight_l) * Phi_s[c].interpolated_value(p.pt)
                + weight_l * Phi_s[1-c].interpolated_value(p.pt);

        p.dPhi_dr = (1.0 - weight_l) * Pi_s[c].interpolated_value(p.pt)
            + weight_l * Pi_s[1-c].interpolated_value(p.pt);

        p.dPhi_dtheta = (1.0 - weight_l) * Phi_s_dtheta[c].interpolated_value(p.pt)
            + weight_l * Phi_s_dtheta[1-c].interpolated_value(p.pt);

        p.dPhi_dphi = (1.0 - weight_l) * Phi_s_dphi[c].interpolated_value(p.pt)
            + weight_l * Phi_s_dphi[1-c].interpolated_value(p.pt);

        p.Omega = (1.0 - weight_l) * Omega_s[c].interpolated_value(p.pt)
            + weight_l * Omega_s[1-c].interpolated_value(p.pt);

        // Ordinary derivatives
        p.dPhi_ddr = (1.0 - weight_l) * Phi_s_ddr[c].interpolated_value(p.pt)
            + weight_l * Phi_s_ddr[1-c].interpolated_value(p.pt);

        p.dPhi_drdtheta = (1.0 - weight_l) * Phi_s_drdtheta[c].interpolated_value(p.pt)
            + weight_l * Phi_s_drdtheta[1-c].interpolated_value(p.pt);
        p.dPhi_drdphi = (1.0 - weight_l) * Phi_s_drdphi[c].interpolated_value(p.pt)
            + weight_l * Phi_s_drdphi[1-c].interpolated_value(p.pt);
        p.dPhi_ddphi = (1.0 - weight_l) * Phi_s_ddphi[c].interpolated_value(p.pt)
            + weight_l * Phi_s_ddphi[1-c].interpolated_value(p.pt);
        p.dPhi_ddtheta = (1.0 - weight_l) * Phi_s_ddtheta[c].interpolated_value(p.pt)
            + weight_l * Phi_s_ddtheta[1-c].interpolated_value(p.pt);
        p.dPhi_dthetadphi = (1.0 - weight_l) * Phi_s_dthetadphi[c].interpolated_value(p.pt)
            + weight_l * Phi_s_dthetadphi[1-c].interpolated_value(p.pt);

        p.Phi_lap = ((1.0-weight_l) * Phi_s_ang_lap[c].interpolated_value(p.pt)
                     + weight_l * Phi_s_ang_lap[1-c].interpolated_value(p.pt))
            / PW2(p.r)
            + (2 * p.dPhi_dr + p.r * p.dPhi_ddr) / p.r ;

        // Adding connection terms to from covariant derivatives
        p.dPhi_ddr += 0;
        p.dPhi_drdtheta -= p.dPhi_dtheta / p.r;
        p.dPhi_drdphi -= p.dPhi_dphi / p.r;
        p.dPhi_ddtheta -= -p.r * p.dPhi_dr;
        p.dPhi_dthetadphi -= p.dPhi_dphi / tan(p.pt.theta);
        p.dPhi_ddphi -= -p.r * PW2(sin(p.pt.theta)) * p.dPhi_dr - cos(p.pt.theta) * sin(p.pt.theta) * p.dPhi_dtheta;
    }


    // rk2 advance from n to n+1
    void time_advance(int n, int c) {

        if(n == NR || to_r(n) > max_tar_r )
        {
            return;
        }

        //update_shell_fields(n);

#pragma omp parallel for
        for(int i = 0; i < n_p; i++)
        {
            if(n > tars_lower_bins[i]) continue;

            Photon p = {0};
            set_photon_values(p, i, n, c);
            double dtau;

            // It is the last step?
            if( n < tars_lower_bins[i])
                dtau = -dr;
            else if(n == tars_lower_bins[i])
                dtau = -(tars[6*i] - p.r);
            else
            {
                cout<<"Wired things might have happened!"<<std::endl;
                throw(-1);
            }
            GEODESIC_APPLY_TO_FIELDS(RK2_ADVANCE_ALL_K1);
            GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_ADVANCE_ALL_K1);

            if(theta_a[i] < 0 || theta_a[i] > M_PI)
                tars_lower_bins[i] = -2;
        }

        c = 1-c;
        update_shell_fields(n+1, c);

        for(int i = 0; i < n_p; i++)
        {
            if(n > tars_lower_bins[i]) continue;
            Photon p = {0};
            double dtau;


            if( n < tars_lower_bins[i])
            {
                dtau = - dr;
                set_photon_values(p, i, n+1, c);
            }
            else if (n == tars_lower_bins[i])
            {
                dtau = -(tars[6*i] - to_r(n));
                set_inter_slice_photon_values(p, i, n, 1-c, fabs(dtau / dr) );
            }
            else
            {
                cout<<"Wired things might have happened!"<<std::endl;
                throw(-1);
            }

            GEODESIC_APPLY_TO_FIELDS(RK2_ADVANCE_ALL_K2);
            GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_ADVANCE_ALL_K2);

            if( n == tars_lower_bins[i])
            {
                set_inter_slice_photon_values(p, i, n, 1-c, fabs(dtau / dr) );
                cal_redshift(p);
            }

            if(theta_a[i] < 0 || theta_a[i] > M_PI)
                tars_lower_bins[i] = -2;
        }

    }

    /*
     * Newton's scheme trying to find root of the function
     * d_theta_t(d_theta), where d_theta_t is the angle difference
     * between target and where rays hit, and d_theta is the difference
     * between your strainght line of sight and your initial shooting direction.
     * Theirfore, the most naive Newton iteration scheme would be
     * d_theta(new) = d_theta(old) - d_theta_t(d_theta) / d_theta_t'(d_theta).
     * d_theta corresponds to array ang_corrs
     */
    void gen_corrs() {

        for(int i = 0; i < n_p; i++)
        {
            if(tars_lower_bins[i] < 0)
            {
                if(tars_lower_bins[i] == -2)
                    diff_s_ang[i] = 0;
                continue;
            }
            double dtheta = theta_a[i] - tars[6*i+1];
            double dphi = phi_a[i] - tars[6*i+2];
            diff_s_ang[i] = fabs(sin(tars[6*i+1]) * dtheta * dphi);
            if(diff_s_ang[i] <= ang_epsilon)
            {
                tars_lower_bins[i] = -1;
                continue;
            }

            double theta_derv = 1.0;
            double phi_derv = 1.0;

            if(fabs(d_diff_ang[2*i]) > 1e-15)
                theta_derv = (dtheta - d_diff_tar_ang[2*i] ) / d_diff_ang[2*i];
            if(fabs(d_diff_ang[2*i+1]) > 1e-15)
                phi_derv = (dphi - d_diff_tar_ang[2*i+1] ) / d_diff_ang[2*i+1];



            ang_corrs[2*i] += - dtheta / theta_derv;
            ang_corrs[2*i+1] += - dphi / phi_derv;

            d_diff_tar_ang[2*i] = dtheta;
            d_diff_tar_ang[2*i+1] = dphi;

            d_diff_ang[2*i] = - dtheta / theta_derv;
            d_diff_ang[2*i+1] = - dphi / phi_derv;
        }
    }
    bool all_on_tars() {
        for(int i = 0; i < n_p; i++)
            if(tars_lower_bins[i] >= 0)
                return false;
        return true;
    }
    void init_rays() {
        update_shell_fields(0, 0);
        for(int i = 0; i < n_p; i++)
        {
            if(tars_lower_bins[i] < 0) continue;

            Photon p = {0};

            theta_a[i] = theta_f[i] = tars[6*i+1] + ang_corrs[2*i];
            phi_a[i] = phi_f[i] = tars[6*i+2] + ang_corrs[2*i+1];

            set_photon_values(p, i, 0, 0);

            k0_a[i] = k0_f[i] = 1.0 * (1-3*p.Phi);
            //k0_a[i] = k0_f[i] = p.a * (1-3*p.Phi);
            DA_a[i] = DA_f[i] = final_r * (1+p.Phi);
            dDAdt_a[i] = dDAdt_f[i] = -(1+2*p.Phi);
            nr_a[i] = nr_f[i] = 1.0;
            ntheta_a[i] = ntheta_f[i] = 0.0;
            nphi_a[i] = nphi_f[i] = 0.0;
        }
    }

    void cal_redshift(Photon &p) {
        double ur = tars[6*p.pid+3];
        double utheta = tars[6*p.pid+4];
        double uphi = tars[6*p.pid+5];
        double q = PW2(ur) + PW2(p.r * utheta) + PW2(p.r * sin(p.theta) * uphi);
        double ni_ui = p.nr * ur + PW2(p.r) * p.ntheta * utheta + PW2(p.r * sin(p.theta)) * p.nphi * uphi;

        z[p.pid] = p.k0 / PW2(p.a) * (1+3*p.Phi) *
            (sqrt(PW2(p.a) * PW2(p.a) * (1-2*p.Phi) * PW2(q) + PW2(p.a) )
             - ni_ui * (1+p.Phi)) ;
        DA_a[p.pid] /= (1 + p.Phi);
    }

    void shoot() {
        int cnt = 0;
        double old_max_s_ang = 1e100;
        auto start = system_clock::now();
        while(all_on_tars() == false && cnt < n_max_shooting)
        {
            init_rays();

            int c = 0;
            // Advancing all particles to the shell that has the same radius with target
            for(int i = 0; i < NR; i++)
            {
                time_advance(i, c);
                c = 1-c;
            }

            // generate angular corret vectors
            gen_corrs();

            double max_s_ang = 0;
            int max_id = -1;
            for(int i = 0; i < n_p; i++)
            {
                if(diff_s_ang[i] > max_s_ang) max_id = i;
                max_s_ang = std::max(max_s_ang, diff_s_ang[i]);
            }
            cout<<"For iteration "<<cnt<<" , the max different solid angle is "<<max_s_ang
                <<", for ray id "<<max_id<<std::endl;

            if(max_s_ang > old_max_s_ang)
                cout<<"Warning!!!, the max angular deviation is increasing!\n";

            old_max_s_ang = max_s_ang;
            cnt++;
        }

        if(cnt == n_max_shooting)
        {
            cout<<"Shooting trial times are more than maximum! It has been forced stop!"<<std::endl;
        }
        auto end   = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        tot_time_con += double(duration.count()) *
            microseconds::period::num / microseconds::period::den;

        cout<<"Total time consumption is "<<tot_time_con<<"s"<<std::endl;
        cout<<"  Time consumption for healpix is "<<hp_time_con<<"s"<<std::endl;
    }

};


#endif
