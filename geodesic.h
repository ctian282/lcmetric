#ifndef GEODESIC_H
#define GEODESIC_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>            
#include <zlib.h>
#include <complex>
#include <vector>
#include "healpix_base.h"
#include "alm_healpix_tools.h"
#include "healpix_map.h"
#include "alm.h"

#include "geodesic_macros.h"

using namespace std::complex_literals;


typedef double real_t;
typedef int idx_t;

typedef std::complex<real_t> cmpx_t;



inline idx_t IDX(idx_t nr, idx_t pix, idx_t NPIX)
{
  return nr*NPIX + pix;
}

struct Photon
{
  idx_t pid;
  pointing pt;
  real_t r, a;
  real_t Phi, dPhi_dr, dPhi_dtheta, dPhi_dphi, dPhi_ddtheta, dPhi_ddphi, dPhi_dthetadphi,
    dPhi_drdtheta, dPhi_drdphi,
    Omega, dPhi_ddr, Phi_lap;

  GEODESIC_APPLY_TO_FIELDS(DECLARE_REAL_T);
  GEODESIC_APPLY_TO_COMPLEX_FIELDS(DECLARE_COMPLEX_T);
};


class _Geodesic
{
  public:

  idx_t NR, NSIDE, NPIX, n_iter, lmax;
  real_t init_r, final_r;
  real_t dr, ang_epsilon;

  idx_t n_p, n_alm_idx;

  
  GEODESIC_APPLY_TO_FIELDS(RK2_FIELDS_ALL_CREATE);
  GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_COMPLEX_FIELDS_ALL_CREATE);
  
  real_t *Phi, *Pi, *Omega, *a, *dPi_dr;
  real_t *tars;
  // initial angular corrections
  real_t *ang_corrs, *diff_s_ang;
  real_t max_tar_r;
  // lower radial bin for tars
  idx_t *tars_lower_bins;


  
  pointing *ang_list;
  
  Healpix_Map <real_t> Phi_s, Pi_s, Omega_s,
    Phi_s_ddr, Phi_s_dtheta, Phi_s_dphi,
    Phi_s_ddtheta, Phi_s_ddphi, Phi_s_dthetadphi, Phi_s_drdtheta, Phi_s_drdphi,
    Phi_s_ang_lap, temp_map;
  Alm< xcomplex< real_t > > temp_lm;
  
  
  
  
  T_Healpix_Base<idx_t> *hp;
  Alm_Base * lm_base;
  idx_t *l_list, *m_list;

  std::ofstream cout;

  inline real_t to_r(idx_t nr)
  {
    return final_r + (init_r - final_r) * (real_t)nr / NR;
  }


  _Geodesic(real_t *Phi_in, real_t *Pi_in, real_t *Omega_in, real_t *dPi_dr_in,
            real_t * a_in, idx_t NR_in, real_t init_r_in, real_t final_r_in):
    cout("debug.txt")
  {
    cout<<"HHHHH\n"<<std::endl;

    return;
  }

  
  _Geodesic(real_t *Phi_in, real_t *Pi_in, real_t *Omega_in, real_t *dPi_dr_in,
            real_t * a_in, idx_t NR_in, real_t init_r_in, real_t final_r_in,
            idx_t NSIDE_in, idx_t n_iter_in= 30, real_t ang_epsilon_in = 1e-6):
    NR(NR_in),
    NSIDE(NSIDE_in),
    NPIX(12*NSIDE*NSIDE),
    n_iter(n_iter_in),
    lmax(2*NSIDE-1),
    init_r(init_r_in),
    final_r(final_r_in),
    dr( (init_r - final_r)/(real_t)NR ),
    ang_epsilon(ang_epsilon_in),
    Phi(Phi_in),
    Pi(Pi_in),
    dPi_dr(dPi_dr_in),
    Omega(Omega_in),
    a(a_in),
    Phi_s(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Pi_s(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Omega_s(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_ddr(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_dtheta(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_dphi(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_ddtheta(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_ddphi(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_dthetadphi(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_drdtheta(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_drdphi(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    Phi_s_ang_lap(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    temp_map(NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE ),
    temp_lm(lmax,lmax),
    cout("debug.txt")
  {
    cout<<"fjljsaf"<<std::endl;

    // Healpix instance
    hp = new T_Healpix_Base<idx_t>
      ( NSIDE, Healpix_Ordering_Scheme::RING, SET_NSIDE );

    // lm_base = new Alm_Base(lmax, lmax);
    // n_alm_idx = lm_base->Num_Alms(lmax, lmax);
    // l_list = new idx_t[n_alm_idx];
    // m_list = new idx_t[n_alm_idx];
    
    // for(int l = 0; l <= lmax; l++)
    // {
    //   for(int m = 0; m <= lmax; m++)
    //   {
    //     l_list[lm_base->index(l,m)] = l;
    //     m_list[lm_base->index(l,m)] = m;
    //   }
    // }
    
    ang_list = new pointing[NPIX];
    for(int i = 0; i < NPIX; i++)
      ang_list[i] = hp->pix2ang(i);
  }
  ~_Geodesic()
  {
    GEODESIC_APPLY_TO_FIELDS(RK2_FIELDS_ALL_DEL);
    GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_FIELDS_ALL_DEL);
  }

  void gen_tars_lower_bins()
  {
    for(int i = 0; i < n_p; i++)
      tars_lower_bins[i] = (idx_t)( (tars[3*i] - final_r) / dr );
  }

  void alloc_par_arrs()
  {
    tars = new real_t [3*n_p]();
    tars_lower_bins = new idx_t [n_p]();
    ang_corrs = new real_t [2*n_p]();
    diff_s_ang = new real_t [n_p]();
  }
  
  // Targets are healpix pixels at distance r
  void init_with_healpix_tars(real_t r)
  {
    max_tar_r = r;
    if(r > init_r)
    {
      std::cout<<"r is too large!"<<std::endl;
      throw(-1);
    }
    n_p = NPIX;
    alloc_par_arrs();
    for(idx_t i = 0; i < NPIX; i++)
    {
      pointing ang = hp->pix2ang(i);

      // real_t x = r * sin(ang.theta) * cos(ang.phi);
      // real_t y = r * sin(ang.theta) * sin(ang.phi);
      // real_t z = r * cos(ang.theta);
      tars[3*i + 0] = r;
      tars[3*i + 1] = ang.theta;
      tars[3*i + 2] = ang.phi;
    }

    gen_tars_lower_bins();
    
    GEODESIC_APPLY_TO_FIELDS_ARGS(RK2_FIELDS_ALL_INIT, n_p);
    GEODESIC_APPLY_TO_COMPLEX_FIELDS_ARGS(RK2_COMPLEX_FIELDS_ALL_INIT, n_p);
    cout<<tars_lower_bins[100]<<" "<<tars[3*10]<<" "<<tars[3*10+1]<<" "<<tars[3*10+2]<<std::endl;
  }


  // 3 things need to be done by this function
  // 1). Extract the healpix array on this shell
  // 2). Convert the alm coeffs of Phi field on this shell
  // 3). Calculating derv1 of Phi field by healpix
  void update_shell_fields(idx_t n)
  {
    for(int i = IDX(n, 0, NPIX); i < IDX(n, 0, NPIX) + NPIX; i++)
    {
      Phi_s[i - IDX(n, 0, NPIX)] = Phi[i];
      Pi_s[i - IDX(n, 0, NPIX)] = Pi[i];
      Omega_s[i - IDX(n, 0, NPIX)] = Omega[i];
      Phi_s_ddr[i - IDX(n, 0, NPIX)] = dPi_dr[i];
    }
    // Calculating first derivatives
    map2alm_iter(Phi_s, temp_lm, n_iter);
    alm2map_der1(temp_lm, temp_map, Phi_s_dtheta, Phi_s_dphi);
    for(idx_t i = 0; i < NPIX; i++)
      Phi_s_dphi[i] *= sin(ang_list[i].theta);

    // Calculating lap
    for(idx_t l = 0; l <= lmax; l++)
      for(idx_t m = 0; m <= l; m++)
      {
        //temp_lm[i] *= -l_list[i] * (l_list[i]+1);
        temp_lm(l, m) *= -(real_t)l *(l+1);
      }
    alm2map(temp_lm, Phi_s_ang_lap);

    // Calculating second and mixing derivatives
    map2alm_iter(Phi_s_dtheta, temp_lm, n_iter);
    alm2map_der1(temp_lm, temp_map, Phi_s_ddtheta, Phi_s_dthetadphi);

    map2alm_iter(Pi_s, temp_lm, n_iter);
    alm2map_der1(temp_lm, temp_map, Phi_s_drdtheta, Phi_s_drdphi);
    for(idx_t i = 0; i < NPIX; i++)
      Phi_s_drdphi[i] *= sin(ang_list[i].theta);

    
    
    // Calculating second and mixing derivatives
    map2alm_iter(Phi_s_dphi, temp_lm, n_iter);
    alm2map_der1(temp_lm, temp_map, Phi_s_dthetadphi, Phi_s_ddphi);
    for(idx_t i = 0; i < NPIX; i++)
      Phi_s_ddphi[i] *= sin(ang_list[i].theta);

  }

  
  /**********************RHS of geodesics***************/

  real_t dtheta_dt(Photon &p)
  {
    return (1.0 + 2.0 * p.Phi ) * p.ntheta;
  }

  real_t dphi_dt(Photon &p)
  {
    return (1.0 + 2.0 * p.Phi ) * p.nphi;
  }

  real_t dk0_dt(Photon &p)
  {
    return -4.0 * (p.dPhi_dr + p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
      - 2.0 * (p.Omega + p.dPhi_dr);
  }

  real_t dnr_dt(Photon &p)
  {
    return 0.0;
  }
  
  real_t dntheta_dt(Photon &p)
  {
    return 2.0 * p.ntheta * ( 1.0 * p.dPhi_dr+ p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
      - 2.0 * (p.dPhi_dtheta / PW2(p.r)  );
  }

  real_t dnphi_dt(Photon &p)
  {
    return 2.0 * p.nphi * ( 1.0 * p.dPhi_dr+ p.ntheta * p.dPhi_dtheta + p.nphi * p.dPhi_dphi)
      - 2.0 * (p.dPhi_dphi / PW2(p.r * sin(p.pt.theta)));

  }
  
  real_t dDA_dt(Photon &p)
  {
    return p.dDAdt;
  }

  real_t ddDAdt_dt(Photon &p)
  {
    return - dk0_dt(p) * p.dDAdt
      + 1.0 * (1 + 2 * p.Phi) *
      (p.nr * p.nr * p.dPhi_ddr + 2.0 * p.nr * p.ntheta * p.dPhi_drdtheta + 2.0 * p.nr * p.nphi * p.dPhi_drdphi
      + p.ntheta * p.ntheta * p.dPhi_ddtheta + 2.0 * p.ntheta * p.nphi * p.dPhi_dthetadphi
      + p.nphi * p.nphi * p.dPhi_ddphi
      - p.Phi_lap ) * p.DA ;
  }


  real_t de1r_dt(Photon &p)
  {
    return p.nr * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }
  real_t de1theta_dt(Photon &p)
  {
    return p.ntheta * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }
  real_t de1phi_dt(Photon &p)
  {
    return p.nphi * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }

  real_t de2r_dt(Photon &p)
  {
    return 1.0 * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }
  real_t de2theta_dt(Photon &p)
  {
    return p.ntheta * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }
  real_t de2phi_dt(Photon &p)
  {
    return p.nphi * 2.0 *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }

  
  real_t dbeta1_dt(Photon &p)
  {
    return 2.0 * p.beta1 * (p.Omega + p.dPhi_dr)
      - (1 + 2 * p.Phi) * p.a *
      (p.e1r * p.dPhi_dr + p.e1theta * p.dPhi_dtheta + p.e1phi * p.dPhi_dphi);
  }

  real_t dbeta2_dt(Photon &p)
  {
    return 2.0 * p.beta2 * (p.Omega + p.dPhi_dr)
      - (1 + 2 * p.Phi) * p.a *
      (p.e2r * p.dPhi_dr + p.e2theta * p.dPhi_dtheta + p.e2phi * p.dPhi_dphi);
  }

  cmpx_t dsigma_dt(Photon &p)
  {
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

  cmpx_t depsilon_dt(Photon &p)
  {
    return 0.0;
  }

  cmpx_t domega_dt(Photon &p)
  {
    return 0.0;
  }

  
  /****************************************************/

  void set_photon_values(Photon &p, idx_t n)
  {
    GEODESIC_APPLY_TO_FIELDS(SET_LOCAL_VALUES);
    GEODESIC_APPLY_TO_COMPLEX_FIELDS(SET_LOCAL_VALUES);

    p.r = to_r(n);
    p.a = a[n];
    
    p.pt = pointing(p.theta, p.phi);
    p.Phi = Phi_s.interpolated_value(p.pt);
    p.dPhi_dr = Pi_s.interpolated_value(p.pt);
    p.dPhi_dtheta = Phi_s_dtheta.interpolated_value(p.pt);
    p.dPhi_dphi = Phi_s_dphi.interpolated_value(p.pt);
    p.Omega = Omega_s.interpolated_value(p.pt);
    
    // Ordinary derivatives
    p.dPhi_ddr = Phi_s_ddr.interpolated_value(p.pt);
    p.dPhi_drdtheta = Phi_s_drdtheta.interpolated_value(p.pt);
    p.dPhi_drdphi = Phi_s_drdphi.interpolated_value(p.pt);
    p.dPhi_ddphi = Phi_s_ddphi.interpolated_value(p.pt);
    p.dPhi_ddtheta = Phi_s_ddtheta.interpolated_value(p.pt);
    p.dPhi_dthetadphi = Phi_s_dthetadphi.interpolated_value(p.pt);

    p.Phi_lap = Phi_s_ang_lap.interpolated_value(p.pt) / PW2(p.r)
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
  void time_advance(idx_t n)
  {
    
    if(n == NR || to_r(n) > max_tar_r )
    {
      return;
    }
    //update_shell_fields(n);
    for(int i = 0; i < n_p; i++)
    {
      if(n > tars_lower_bins[i]) continue;
      
      Photon p;
      set_photon_values(p, n);
      real_t dtau;
      if( n < tars_lower_bins[i])
        dtau = -dr;
      else if(n == tars_lower_bins[i])
      {
        dtau = -(tars[3*i] - p.r);
      }
      else
      {
        cout<<"Wired things might have happened!"<<std::endl;
        throw(-1);
      }
      GEODESIC_APPLY_TO_FIELDS(RK2_ADVANCE_ALL_K1);
      GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_ADVANCE_ALL_K1);

    }

    update_shell_fields(n+1);

    for(int i = 0; i < n_p; i++)
    {
      if(n >= tars_lower_bins[i]) continue;
      Photon p;
      real_t dtau = -dr;
      set_photon_values(p, n+1);
      GEODESIC_APPLY_TO_FIELDS(RK2_ADVANCE_ALL_K2);
      GEODESIC_APPLY_TO_COMPLEX_FIELDS(RK2_ADVANCE_ALL_K2);
    }
    // cout<<"Done with first ta "<<dDAdt_c[0]<<std::endl;
    // throw(-1);

    // Photon p;
    // set_photon_values(p, n+1);
    

    // cout<<k0_c[0]<<" "<<p.dPhi_dr<<" "<<p.dPhi_dtheta<<" "<<p.dPhi_dphi
    //     <<" "<<p.Omega<<" "<<dk0_dt(p)<<" "<<k0_a[0]<<std::endl;
    
  }

  void gen_corrs()
  {
    for(int i = 0; i < n_p; i++)
    {
      real_t dtheta = theta_a[i] - tars[3*i+1];
      real_t dphi = phi_a[i] - tars[3*i+2];
      diff_s_ang[i] = fabs(sin(tars[3*i+1]) * dtheta * dphi);
      if(diff_s_ang[i] <= ang_epsilon)
      {
        tars_lower_bins[i] = -1;
        continue;
      }
      ang_corrs[2*i] += - dtheta;
      ang_corrs[2*i+1] += - dphi;
    }
  }
  bool all_on_tars()
  {
    for(int i = 0; i < n_p; i++)
      if(tars_lower_bins[i] >= 0)
        return false;
    return true;
  }
  void init_rays()
  {

    update_shell_fields(0);

    for(int i = 0; i < n_p; i++)
    {
      if(tars_lower_bins[i] < 0) continue;

      Photon p;
      
      set_photon_values(p, 0);
      
      theta_a[i] = theta_f[i] = tars[3*i+1] + ang_corrs[2*i];
      phi_a[i] = phi_f[i] = tars[3*i+2] + ang_corrs[2*i+1];
      k0_a[i] = k0_f[i] = a[0] * (1-3*p.Phi);
      DA_a[i] = DA_f[i] = final_r * (1+p.Phi);
      dDAdt_a[i] = dDAdt_f[i] = -(1+2*p.Phi);
      nr_a[i] = nr_f[i] = 1.0;
      ntheta_a[i] = ntheta_f[i] = 0.0;
      nphi_a[i] = nphi_f[i] = 0.0;
    }
  }
  void shoot()
  {
    int cnt = 0;
    real_t old_max_s_ang = 1e100;
    while(all_on_tars() == false)
    {
      init_rays();
      // Advancing all particles to the shell that has the same radius with target
      for(int i = 0; i < NR; i++)
        time_advance(i);

      cout<<"lower bin is "<<tars_lower_bins[0]<<" NR is "<<NR<<std::endl;

      gen_corrs();

      real_t max_s_ang = 0;
      idx_t max_id = -1;
      for(int i = 0; i < n_p; i++)
      {
        if(diff_s_ang[i] > max_s_ang) max_id = i;
        max_s_ang = std::max(max_s_ang, diff_s_ang[i]);
      }
      // cout<<"378 "<<diff_s_ang[378]<<" "<<ang_corrs[2*378]<<" "<<ang_corrs[2*378+1]
      //     <<std::endl;
      cout<<"For iteration "<<cnt<<" , the max different solid angle is "<<max_s_ang
          <<", for ray id "<<max_id<<std::endl;

      if(max_s_ang > old_max_s_ang)
        cout<<"Warning!!!, the max angular deviation is increasing!\n";
      
      old_max_s_ang = max_s_ang;
      cnt++;
    }
    
  }
};


#endif
