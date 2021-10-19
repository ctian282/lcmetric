#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <zlib.h>
#include "healpix_base.h"


typedef long long idx_t;


#define CHI2CHIBIN(r, dr) std::floor ((r) / (dr));

inline idx_t IDX(int nr, int pix, int NPIX)
{
    return (idx_t)nr*NPIX + (idx_t)pix;
}

template<typename T>
void kappa_deposit(T *particles, double* a, double *origin,
                   T *kappa1, T *kappa2, idx_t nparticles,
                   double max_r, double min_r, int NR, int NSIDE, double shift)
{
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    double dr = (max_r - min_r) / NR;

    int NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++){
        double x = particles[p*6 + 0] - origin[0];
        double y = particles[p*6 + 1] - origin[1];
        double z = particles[p*6 + 2] - origin[2];

        double r = std::sqrt(x*x + y*y + z*z) + shift;
        // get chi, theta, phi
        double theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        double phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        double dc = (r-min_r) / dr - (double)ic;

        double tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        auto pix = HP->ang2pix(ptg);

        double ap = a[ic+1] * tc + a[icp+1] * dc;
        if(dc <= 0.5 && ic >= 0){
#pragma omp atomic
            kappa1[IDX(ic, pix, NPIX)] += 1.0 / ap / r;
#pragma omp atomic
            kappa2[IDX(ic, pix, NPIX)] += 1.0 / ap;
        }
        else if(dc > 0.5){
#pragma omp atomic
            kappa1[IDX(icp, pix, NPIX)] += 1.0 / ap / r;
#pragma omp atomic
            kappa2[IDX(icp, pix, NPIX)] += 1.0 / ap;
        }
    }

}

template<typename T>
void NGP_deposit(T *particles, double *origin, T *delta,
                 T *vw, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    double dr = (max_r - min_r) / NR;

    int NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        double x = particles[p*6 + 0] - origin[0];
        double y = particles[p*6 + 1] - origin[1];
        double z = particles[p*6 + 2] - origin[2];
        double vx = particles[p*6 + 3];
        double vy = particles[p*6 + 4];
        double vz = particles[p*6 + 5];

        double r = std::sqrt(x*x + y*y + z*z) + shift;
        // get chi, theta, phi
        double theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        double phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        double vr = (vx * x + vy * y + vz * z) / r;

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        double chi_min = r - dr/2.0;
        double chi_max = r + dr/2.0;
        double expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((double)      \
                                                           NPIX);

        double dc = (r-min_r) / dr - (double)ic;

        double tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        // auto pix = fix_arr<int, 4>();
        // auto wgt = fix_arr<double, 4>();
        // HP->get_interpol(ptg, pix, wgt);
        auto pix = HP->ang2pix(ptg);
        //CIC deposit
        if( ic >= 0)
        {
            //for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(ic, pix, NPIX)] += 1.0/expected_counts*tc;
#pragma omp atomic
                delta[IDX(icp, pix, NPIX)] += 1.0/expected_counts*dc;

#pragma omp atomic
                counts[IDX(ic, pix, NPIX)] += tc;
#pragma omp atomic
                counts[IDX(icp, pix, NPIX)] += dc;

#pragma omp atomic
                vw[IDX(ic, pix, NPIX)] += vr*tc;
#pragma omp atomic
                vw[IDX(icp, pix, NPIX)] += vr*dc;
            }
        }
        else
        {
            //for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(icp, pix, NPIX)] += 1.0/expected_counts*dc;

#pragma omp atomic
                counts[IDX(icp, pix, NPIX)] += dc;

#pragma omp atomic
                vw[IDX(icp, pix, NPIX)] += vr*dc;
            }
        }
    }

}

template<typename T>
void CIC_deposit_with_wgt(
    T *particles, double *origin, T *delta, T *vw, T *counts,
    idx_t nparticles, double count_density,
    double max_r, double min_r, int NR, int NSIDE, T *weight, double shift)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>

        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );
    double dr = (max_r - min_r) / NR;

    int NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        double x = particles[p*7 + 0] - origin[0];
        double y = particles[p*7 + 1] - origin[1];
        double z = particles[p*7 + 2] - origin[2];
        double vx = particles[p*7 + 3];
        double vy = particles[p*7 + 4];
        double vz = particles[p*7 + 5];

        double val = particles[p*7 + 6];

        double r = std::sqrt(x*x + y*y + z*z) + shift;
        // get chi, theta, phi
        double theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        double phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        double vr = (vx * x + vy * y + vz * z) / r;

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        double chi_min = r - dr/2.0;
        double chi_max = r + dr/2.0;
        double expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((double) NPIX);

        double dc = (r-min_r) / dr - (double)ic;

        double tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        auto pix = fix_arr<int, 4>();
        auto wgt = fix_arr<double, 4>();
        HP->get_interpol(ptg, pix, wgt);

        //CIC deposit
        if( ic >= 0)
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(ic, pix[i], NPIX)] += 1.0/expected_counts*tc*wgt[i];
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(ic, pix[i], NPIX)] += tc*wgt[i];
#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(ic, pix[i], NPIX)] += vr*tc*wgt[i];
#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];

#pragma omp atomic
                weight[IDX(ic, pix[i], NPIX)] += val*tc*wgt[i];
#pragma omp atomic
                weight[IDX(icp, pix[i], NPIX)] += val*dc*wgt[i];

            }
        }
        else
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];

#pragma omp atomic
                weight[IDX(icp, pix[i], NPIX)] += val*dc*wgt[i];

            }
        }
    }
}

template<typename T>
void CIC_deposit(T *particles, double *origin, T *delta,
                 T *vw, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    double dr = (max_r - min_r) / NR;

    int NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        double x = particles[p*6 + 0] - origin[0];
        double y = particles[p*6 + 1] - origin[1];
        double z = particles[p*6 + 2] - origin[2];
        double vx = particles[p*6 + 3];
        double vy = particles[p*6 + 4];
        double vz = particles[p*6 + 5];

        double r = std::sqrt(x*x + y*y + z*z) + shift;
        // get chi, theta, phi
        double theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        double phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        double vr = (vx * x + vy * y + vz * z) / r;


        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        double chi_min = r - dr/2.0;
        double chi_max = r + dr/2.0;
        double expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((double)      \
                                                           NPIX);

        double dc = (r-min_r) / dr - (double)ic;

        double tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        auto pix = fix_arr<int, 4>();
        auto wgt = fix_arr<double, 4>();
        HP->get_interpol(ptg, pix, wgt);

        //CIC deposit
        if( ic >= 0)
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(ic, pix[i], NPIX)] += 1.0/expected_counts*tc*wgt[i];
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(ic, pix[i], NPIX)] += tc*wgt[i];
#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(ic, pix[i], NPIX)] += vr*tc*wgt[i];
#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];
            }
        }
        else
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];
            }
        }
    }

}

template<typename T>
void CIC_deposit_ang(T *particles, double *origin, T *delta,
                     T *vw, T *vtheta, T*vphi, T *counts,
                 idx_t nparticles, double count_density,
                 double max_r, double min_r, int NR, int NSIDE, double shift)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    double dr = (max_r - min_r) / NR;

    int NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        double x = particles[p*6 + 0] - origin[0];
        double y = particles[p*6 + 1] - origin[1];
        double z = particles[p*6 + 2] - origin[2];
        double vx = particles[p*6 + 3];
        double vy = particles[p*6 + 4];
        double vz = particles[p*6 + 5];

        double r = std::sqrt(x*x + y*y + z*z) + shift;
        // get chi, theta, phi
        double theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        double phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        double vr = (vx * x + vy * y + vz * z) / r;
        double vt = (- r * r * vz + z * (vx * x + vy * y + vz * z) ) / (r*r * sqrt(r * r - z * z));
        double vp = vy * x - vx * y / (x * x + y * y);

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        double chi_min = r - dr/2.0;
        double chi_max = r + dr/2.0;
        double expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((double)      \
                                                           NPIX);

        double dc = (r-min_r) / dr - (double)ic;

        double tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        auto pix = fix_arr<int, 4>();
        auto wgt = fix_arr<double, 4>();
        HP->get_interpol(ptg, pix, wgt);

        //CIC deposit
        if( ic >= 0)
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(ic, pix[i], NPIX)] += 1.0/expected_counts*tc*wgt[i];
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(ic, pix[i], NPIX)] += tc*wgt[i];
#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(ic, pix[i], NPIX)] += vr*tc*wgt[i];
#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];
#pragma omp atomic
                vtheta[IDX(ic, pix[i], NPIX)] += vt*tc*wgt[i];
#pragma omp atomic
                vtheta[IDX(icp, pix[i], NPIX)] += vt*dc*wgt[i];
#pragma omp atomic
                vphi[IDX(ic, pix[i], NPIX)] += vp*tc*wgt[i];
#pragma omp atomic
                vphi[IDX(icp, pix[i], NPIX)] += vp*dc*wgt[i];
            }
        }
        else
        {
            for(int i = 0 ; i < 4; i++)
            {
#pragma omp atomic
                delta[IDX(icp, pix[i], NPIX)] += 1.0/expected_counts*dc*wgt[i];

#pragma omp atomic
                counts[IDX(icp, pix[i], NPIX)] += dc*wgt[i];

#pragma omp atomic
                vw[IDX(icp, pix[i], NPIX)] += vr*dc*wgt[i];
#pragma omp atomic
                vtheta[IDX(icp, pix[i], NPIX)] += vt*dc*wgt[i];
#pragma omp atomic
                vphi[IDX(icp, pix[i], NPIX)] += vp*dc*wgt[i];

            }
        }
    }

}
