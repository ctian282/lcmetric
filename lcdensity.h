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
typedef int idx_t;

class _DensFromSnaps
{
 public:

  idx_t n_part, n_obs;

  idx_t max_move_x, max_move_y, max_move_z;

  real_t init_r;
  // mapping particles' ID to positions
  real_t *id2pos, *obs;

  bool is_first_snap;

  bool *is_out;

  std::vector<std::vector<real_t>> lc_p;


  std::ofstream cout;

 _DensFromSnaps(idx_t n_part_in, idx_t n_bos_in, real_t * obs_in, real_t init_r_in,
                real_t *box):
    n_part(n_part_in),
      init_r(init_r_in),
      n_obs(n_obs_in),
      obs(obs_in),
      lc_p(n_obs),
      is_first_snap(true),
      cout("dens_debug.txt")
    {
      id2pos = new real_t[3*(n_part + 1)];
      is_out = new bool[n_part];

      max_move_x = floor(init_r / box[0]);
      max_move_y = floor(init_r / box[1]);
      max_move_z = floor(init_r / box[2]);

    }

  ~_DensFromSnaps()
    {
      delete [] pos;
    }


  void advance_snap(real_t *pos, real_t *vel real_t ids, real_t lc_r)
  {
    for(int i = 0; i < n_part; i ++){

      for(int mx = -max_move_x; mx <= max_move_x; mx++){
        for(int my = - max_move_y; my <= max_move_y; my++){
          for(int mz = -max_move_z; mz <= max_move_z; mz++){

            real_t old_x = id2pos[3 * ids[i] + 0] + box[0] * mx;
            real_t old_y = id2pos[3 * ids[i] + 1] + box[1] * my;
            real_t old_z = id2pos[3 * ids[i] + 2] + box[2] * mz;
            real_t old_r = sqrt(PW2(old_x) + PW2(old_y) + PW2(old_z));

            // if alread outside the lightcone
            if(old_r > lc_r) continue;

            real_t new_x = pos[3 * i] + box[0] * mx;
            real_t new_y = pos[3 * i + 1] + box[1] * yz;
            real_t new_z = pos[3 * i + 2] + box[2] * mz;
            real_t new_r = sqrt(PW2(new_x) + PW2(new_y) + PW2(new_z));

            if(new_r < lc_r) continue;


          }
        }
      }

    }
  }

  void update_pos_map(real_t * pos, real_t *ids)
  {
    for(int i = 0; i < n_part; i++){
      if( ids[i] > n_part){
        cout<<"ERROR! The id is tool large!";
        throw(-1);
      }
      id2pos[3* ids[i] ] = pos[3 * i];
      id2pos[3* ids[i] + 1] = pos[3 * i + 1];
      id2pos[3* ids[i] + 2] = pos[3 * i + 2];
    }

  }

  void proc_snap(real_t * pos, real_t * vel, real_t *ids, real_t dtau, real_t lc_r)
  {
    if(is_first_snap == true){
      update_pos_map(real_t* pos,real_t* ids);
      is_first_snap = false;
    }
    else{
      
    }
  }

};


#endif
