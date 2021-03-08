#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>            
#include <zlib.h>
#include <complex>
#include <vector>
#include "/usr/include/healpix_cxx/healpix_base.h"

#include "/usr/include/healpix_cxx/healpix_map.h"


int main()
{
  Healpix_Map<double> Phi_s[2];
  //auto *Phi_s = new Healpix_Map<double>[2];
  for(int i = 0; i < 2; i ++)
    Phi_s[i].SetNside(8, RING);

  pointing pt(3.14,20);
  Phi_s[0].interpolated_value(pt);
  
  std::vector<int> *t = new std::vector<int> [2];
  for(int i = 0; i < 2; i++)
    std::cout<<(t[i].size())<<std::endl;
  pointing * ang_list;
  int NSIDE = 8;
  int NPIX = 12*NSIDE*NSIDE;
  Phi_s[0][10] = 999;
  return 0;
}
