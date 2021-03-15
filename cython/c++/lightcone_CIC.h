#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <zlib.h>
#include "healpix_base.h"


typedef double real_t;
typedef long long idx_t;

inline int CHI2CHIBIN(real_t r, real_t dr)
{
  return std::floor ((r) / dr);
}

inline idx_t IDX(int nr, int pix, int NPIX)
{
    return (idx_t)nr*NPIX + pix;
}

void kappa_deposit(real_t *particles, real_t *a, real_t *origin,
                   real_t *kappa1, real_t *kappa2, idx_t nparticles,
                   real_t max_r, real_t min_r, int NR, int NSIDE)
{
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    real_t dr = (max_r - min_r) / NR;

    idx_t NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++){
        real_t x = particles[p*6 + 0] - origin[0];
        real_t y = particles[p*6 + 1] - origin[1];
        real_t z = particles[p*6 + 2] - origin[2];

        real_t r = std::sqrt(x*x + y*y + z*z);
        // get chi, theta, phi
        real_t theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        real_t phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        real_t dc = (r-min_r) / dr - (real_t)ic;

        real_t tc = 1.0 - dc;

        pointing ptg = pointing(theta, phi);
        auto pix = HP->ang2pix(ptg);

        real_t ap = a[ic+1] * tc + a[icp+1] * dc;
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

void CIC_deposit(real_t *particles, real_t *origin, real_t *delta,
                 real_t *vw, real_t *counts,
                 idx_t nparticles, real_t count_density,
                 real_t max_r, real_t min_r, int NR, int NSIDE, int vx_is_weight)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>
        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    real_t dr = (max_r - min_r) / NR;

    idx_t NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        real_t x = particles[p*6 + 0] - origin[0];
        real_t y = particles[p*6 + 1] - origin[1];
        real_t z = particles[p*6 + 2] - origin[2];
        real_t vx = particles[p*6 + 3];
        real_t vy = particles[p*6 + 4];
        real_t vz = particles[p*6 + 5];

        real_t r = std::sqrt(x*x + y*y + z*z);
        // get chi, theta, phi
        real_t theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        real_t phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        real_t vr = (vx * x + vy * y + vz * z) / r;

        if(vx_is_weight > 0)
        {
            vr = vx;
        }

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        real_t chi_min = r - dr/2.0;
        real_t chi_max = r + dr/2.0;
        real_t expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((real_t)\
                                                           NPIX);

        real_t dc = (r-min_r) / dr - (real_t)ic;

        real_t tc = 1.0 - dc;

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

void CIC_deposit_with_wgt(
    real_t *particles, real_t *origin, real_t *delta, real_t *vw, real_t *counts,
    idx_t nparticles, real_t count_density,
    real_t max_r, real_t min_r, int NR, int NSIDE, real_t *weight)
{
    // Healpix instance
    T_Healpix_Base<int> * HP = new T_Healpix_Base<int>

        ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );
    real_t dr = (max_r - min_r) / NR;

    idx_t NPIX = 12*NSIDE*NSIDE;

#pragma omp parallel for
    for(idx_t p = 0; p < nparticles; p++)
    {
        real_t x = particles[p*7 + 0] - origin[0];
        real_t y = particles[p*7 + 1] - origin[1];
        real_t z = particles[p*7 + 2] - origin[2];
        real_t vx = particles[p*7 + 3];
        real_t vy = particles[p*7 + 4];
        real_t vz = particles[p*7 + 5];

        real_t val = particles[p*7 + 6];

        real_t r = std::sqrt(x*x + y*y + z*z);
        // get chi, theta, phi
        real_t theta = std::acos(z/r);
        if(theta < -0.1) throw(-1);
        if(theta >= M_PI) theta = M_PI - 1.0e-6; // correct for floating point error
        real_t phi = std::atan2(y, x);
        if(phi <= 0.0) { phi = 2.0*M_PI + phi; } // atan2 in range (-pi,pi) -> (0,2pi)
        if(phi >= 2.0*M_PI) phi = 2.0*M_PI - 1.0e-6;

        real_t vr = (vx * x + vy * y + vz * z) / r;

        int ic = CHI2CHIBIN(r-min_r, dr);
        int icp = ic+1;

        if(icp >= NR + 2 || ic < -1) continue;

        real_t chi_min = r - dr/2.0;
        real_t chi_max = r + dr/2.0;
        real_t expected_counts = count_density * 4.0/3.0*M_PI*(
            std::pow(chi_max,3) - std::pow(chi_min,3) ) / ((real_t) NPIX);

        real_t dc = (r-min_r) / dr - (real_t)ic;

        real_t tc = 1.0 - dc;

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
