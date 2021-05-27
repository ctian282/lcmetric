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


typedef long long idx_t;

template<class T, class IT> class _DensFromSnaps
{
public:

    IT n_tot_part;
    int n_obs;

    int max_move_x, max_move_y, max_move_z;

    double init_r;
    // mapping particles' ID to positions
    T *id2pos, *id2vel;
    double *obs, *box;

    bool *is_out;

    std::vector<T> lc_p;


    std::ofstream cout;

    _DensFromSnaps(IT n_tot_part_in, double * obs_in, double init_r_in,
                   double *box_in):n_tot_part(n_tot_part_in),
                                   init_r(init_r_in),
                                   obs(obs_in),
                                   cout("dens_debug.txt") {
        id2pos = new T[3*(n_tot_part + 1)];
        id2vel = new T[3*(n_tot_part + 1)];

        box = new double[3];
        for(int i = 0; i < 3; i++) box[i] = box_in[i];
    }

    ~_DensFromSnaps(){

        delete [] id2pos;
        delete [] id2vel;
    }

    void clear_lc() {
        lc_p.clear();
    }

    void advance_snap(T *pos, T *vel, IT *ids, double lc_r,
                      double tau, double dtau, T a_dot, T a, IT np) {

        max_move_x = ceil(2*lc_r / box[0]);
        max_move_y = ceil(2*lc_r / box[1]);
        max_move_z = ceil(2*lc_r / box[2]);

#pragma omp parallel for
        for(IT i = 0; i < np; i ++){
            for(int mx = -max_move_x; mx <= max_move_x; mx++){
                for(int my = - max_move_y; my <= max_move_y; my++){
                    for(int mz = -max_move_z; mz <= max_move_z; mz++){
                        double x = id2pos[3 * ids[i] + 0] + box[0] * mx - obs[0];
                        double y = id2pos[3 * ids[i] + 1] + box[1] * my - obs[1];
                        double z = id2pos[3 * ids[i] + 2] + box[2] * mz - obs[2];
                        double r = sqrt(PW2(x) + PW2(y) + PW2(z));

                        // if alread outside the lightcone
                        if(r > lc_r) continue;

                        double next_x = pos[3 * i + 0] + box[0] * mx - obs[0];
                        double next_y = pos[3 * i + 1] + box[1] * my - obs[1];
                        double next_z = pos[3 * i + 2] + box[2] * mz - obs[2];
                        double next_r = sqrt(PW2(next_x) + PW2(next_y) + PW2(next_z));

                        if(next_r < lc_r - dtau) continue;

                        double vx = id2vel[3 * ids[i] + 0];
                        double vy = id2vel[3 * ids[i] + 1];
                        double vz = id2vel[3 * ids[i] + 2];

                        // double vx = (next_x - x ) / dtau;
                        // double vy = (next_y - y ) / dtau;
                        // double vz = (next_z - z ) / dtau;

                        double next_vx = vel[3 * i + 0] ;
                        double next_vy = vel[3 * i + 1] ;
                        double next_vz = vel[3 * i + 2] ;

                        double ax = (next_vx - vx) / dtau;
                        double ay = (next_vy - vy) / dtau;
                        double az = (next_vz - vz) / dtau;

                        // double alpha = + 0.5 *
                        //     (PW2(x * vx + y * vy + z * vz) / PW3(r) -
                        //      (vx * vx + vy * vy + vz * vz) / r );

                        double alpha = + 0.5 *
                            (PW2(x * vx + y * vy + z * vz) / PW3(r) -
                             (vx * vx + vy * vy + vz * vz) / r -
                             (ax * x + ay * y + az * z) / r) ;

                        double beta = -1.0 - (x * vx + y * vy + z * vz) / r;
                        double gamma = -(tau) - r;

                        double dt = -gamma / beta - alpha *  PW2(gamma) / PW3(beta);
                        if(PW2(beta) - 4.0 * alpha * gamma >= 0 ){
                            double dt1 = (-beta + sqrt(PW2(beta) - 4.0 * alpha * gamma))
                                / (2.0 * alpha);
                            double dt2 = (-beta - sqrt(PW2(beta) - 4.0 * alpha * gamma))
                                / (2.0 * alpha);
                            if(dt1 > 0 && dt1 < dtau)
                                dt = dt1;
                            else if(dt2 > 0 && dt2 < dtau)
                                dt = dt2;
                        }

                        if(dt < 0 || dt > dtau) continue;


                        T lc_x = x + vx * dt;
                        T lc_y = y + vy * dt;
                        T lc_z = z + vz * dt;
                        std::vector<T> vec{lc_x, lc_y, lc_z, (T)vx, (T)vy, (T)vz};

#pragma omp critical
                        std::copy(begin(vec), end(vec), std::back_inserter(lc_p));
                    }
                }
            }

        }
    }

    void update_pos_map(T * pos, T * vel, IT *ids, IT np){

#pragma omp parallel for
        for(IT i = 0; i < np; i++){
            if( ids[i] > n_tot_part){
                cout<<"ERROR! The id is tool large!";
                throw(-1);
            }
            id2pos[3* ids[i] ] = pos[3 * i];
            id2pos[3* ids[i] + 1] = pos[3 * i + 1];
            id2pos[3* ids[i] + 2] = pos[3 * i + 2];

            id2vel[3* ids[i] ] = vel[3 * i];
            id2vel[3* ids[i] + 1] = vel[3 * i + 1];
            id2vel[3* ids[i] + 2] = vel[3 * i + 2];

        }

    }

    void proc_snap(T * pos, T * vel, IT *ids, double tau,
                   double dtau, T a, T H, bool is_first_snap) {
        if(is_first_snap == true){
            update_pos_map(pos, vel, ids, n_tot_part);
        }
        else{
            advance_snap(pos, vel, ids, -tau + dtau, tau - dtau, dtau, a * H, a, n_tot_part);
            update_pos_map(pos, vel, ids, n_tot_part);
        }
    }

    void proc_snap_chunk(T * pos, T * vel, IT *ids, double tau,
                         double dtau, T a, T H, IT np, bool is_first_snap) {
        if(is_first_snap == true){
            update_pos_map(pos, vel, ids, np);
        }
        else{
            advance_snap(pos, vel, ids, -tau + dtau, tau - dtau, dtau, a * H, a, np);
            update_pos_map(pos, vel, ids, np);
        }
    }

};


#endif
