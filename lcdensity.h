#ifndef LCDENSITY_H
#define LCDENSITY_H

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


typedef double real_t;
typedef long long idx_t;

class _DensFromSnaps
{
public:

  idx_t n_part;
  int n_obs;

  int max_move_x, max_move_y, max_move_z;

  real_t init_r;
  // mapping particles' ID to positions
  real_t *id2pos, *obs, *box;

  bool is_first_snap;

  bool *is_out;

  std::vector<real_t> lc_p;


  std::ofstream cout;

  _DensFromSnaps(idx_t n_part_in, real_t * obs_in, real_t init_r_in,
                 real_t *box_in):n_part(n_part_in),
                              init_r(init_r_in),
                              obs(obs_in),
                              is_first_snap(true),
                              cout("dens_debug.txt")
    {
      //id2pos = new real_t[3*(n_part + 1)];
      //is_out = new bool[n_part];

      box = new real_t[3];
      for(int i = 0; i < 3; i++) box[i] = box_in[i];

/*       max_move_x = floor(2*init_r / box[0]);
      max_move_y = floor(2*init_r / box[1]);
      max_move_z = floor(2*init_r / box[2]);
 */
      cout<<max_move_x<<std::endl;
    }

  ~_DensFromSnaps()
  {
    //delete [] id2pos;
  }

  void clear_lc()
  {
    lc_p.clear();
  }

  void advance_snap(real_t *pos, real_t *vel, idx_t *ids, real_t lc_r,
                    real_t tau, real_t dtau, real_t a_dot, real_t a)
  {
    for(idx_t i = 0; i < n_part; i ++){

      max_move_x = floor(2*lc_r / box[0]);
      max_move_y = floor(2*lc_r / box[1]);
      max_move_z = floor(2*lc_r / box[2]);

      for(int mx = -max_move_x; mx <= max_move_x; mx++){
        for(int my = - max_move_y; my <= max_move_y; my++){
          for(int mz = -max_move_z; mz <= max_move_z; mz++){

            real_t x = pos[3 * i + 0] + box[0] * mx - obs[0];
            real_t y = pos[3 * i + 1] + box[1] * my - obs[1];
            real_t z = pos[3 * i + 2] + box[2] * mz - obs[2];
            real_t r = sqrt(PW2(x) + PW2(y) + PW2(z));

            // if alread outside the lightcone
            if(r > lc_r) continue;

            // if(next_r < lc_r) continue;
            real_t vx = vel[3 * i + 0];
            real_t vy = vel[3 * i + 1];
            real_t vz = vel[3 * i + 2];
            // real_t alpha = a_dot / (2.0 * PW2(a)) + 0.5 *
            //   (PW2(x * vx + y * vy + z * vz) / PW3(r) -
            //    (vx * vx + vy * vy + vz * vz) / r );
            real_t alpha = + 0.5 *
                (PW2(x * vx + y * vy + z * vz) / PW3(r) -
                 (vx * vx + vy * vy + vz * vz) / r );
            // real_t beta = -1.0 / a - (x * vx + y * vy + z * vz) / r;
            // real_t gamma = -(tau + dtau) + dtau / a
            //   - a_dot * PW2(dtau) / (2 * PW2(a)) - r;

            real_t beta = -1.0 - (x * vx + y * vy + z * vz) / r;
            real_t gamma = -(tau) - r;

            real_t dt1 = -gamma / beta - alpha * PW2(gamma) / PW3(beta);

            real_t gamma2 = -tau - r;

            // real_t dt2 = -gamma2 / beta - alpha * PW2(gamma2) / PW3(beta);
            // real_t w = (dt1 + dt2) / 2.0 / dtau;
            // real_t dt = dt1 * w + dt2 * (1.0 - w);
            real_t dt = dt1;
            if(dt < 0 || dt > dtau) continue;
            // if(dt1 > 0 && dt1 < dtau)
            //   dt = dt1;
            // else if(dt2 > 0 && dt2 < dtau)
            //   dt = dt2;
            // else{
            //   cout<<"Cannot find proper dt!"<<std::endl;
            //   cout<<r<<" "<<" "<<" "<<alpha<<" "<<beta<<" "<<gamma
            //       <<" "<<dt1<<" "<<dt2<<std::endl;
            //   throw(-1);
            // }

            real_t lc_x = x + vx * dt;
            real_t lc_y = y + vy * dt;
            real_t lc_z = z + vz * dt;
            std::vector<real_t> vec{lc_x, lc_y, lc_z, vx, vy, vz};
            std::copy(begin(vec), end(vec), std::back_inserter(lc_p));
          }
        }
      }

    }
  }

  void update_pos_map(real_t * pos, idx_t *ids)
  {
    for(idx_t i = 0; i < n_part; i++){
      if( ids[i] > n_part){
        cout<<"ERROR! The id is tool large!";
        throw(-1);
      }
      id2pos[3* ids[i] ] = pos[3 * i];
      id2pos[3* ids[i] + 1] = pos[3 * i + 1];
      id2pos[3* ids[i] + 2] = pos[3 * i + 2];
    }

  }

  void proc_snap(real_t * pos, real_t * vel, idx_t *ids, real_t tau,
                 real_t dtau, real_t a, real_t H)
  {
    // if(is_first_snap == true){
    //   update_pos_map(pos, ids);
    //   is_first_snap = false;
    // }
    // else{
    //   advance_snap(pos, vel, ids, -tau + dtau, tau - dtau, dtau, a * H, a);
    //   update_pos_map(pos, ids);
    // }
    advance_snap(pos, vel, ids, -tau, tau, dtau, a * H, a);
  }

};


#endif
