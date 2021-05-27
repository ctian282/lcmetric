#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <zlib.h>

#include <omp.h>
typedef double real_t;
typedef int idx_t;



#define IDX(i, j, k, nx, ny, nz) ( (long long)( ((i)%(nx) + nx)% nx ) * (ny) * (nz) + (long long)( ( (j)%(ny) + ny)%ny ) * (nz) + (long long)(((k)%(nz) + nz)%nz ) )

template<typename T>
void _interp(T *d, double *dx, int nx, int ny, int nz,
             T *x_list, int ns, T *res)
{
  #pragma omp parallel for
  for(int n = 0; n < ns; n++)
  {
    idx_t lx = std::floor(x_list[3*n + 0] / dx[0]);
    res[n] = 0;
    for(idx_t i = lx; i <= lx+1; i++)
    {
      real_t tempj = 0;
      idx_t ly = std::floor(x_list[3*n + 1] / dx[1]);
      for(idx_t j = ly; j <= ly+1; j++)
      {
        real_t tempk = 0;
        idx_t lz = std::floor(x_list[3*n + 2] / dx[2]);
        for(idx_t k = lz; k <= lz+1; k++)
        {
          tempk += d[IDX(i, j, k, nx, ny, nz)]
            * (1.0 - std::fabs(x_list[3*n+2] - (real_t)k * dx[2]) / dx[2] );
        }
        tempj += tempk * (1.0 - std::fabs(x_list[3*n+1] - (real_t)j * dx[1]) / dx[1] );
      }
      res[n] += tempj * (1.0 - std::fabs(x_list[3*n+0] - (real_t)i * dx[0]) / dx[0] );
    }
  }
}
