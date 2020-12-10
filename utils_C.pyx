import pyximport
pyximport.install(reload_support=True)

import numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI


cdef derv1(double c, double l, double r, double dx):
    return (- 0.5 * l  + 0.5 * r ) / dx


cdef derv2(double c, double l, double r, double dx):
    return (l  +  r - 2.0 * c ) / (dx * dx)


# 
def getlr(int idx, int l, int r, double dx, int nside):
    cdef int lower_b, lower_idx, temp_idx
    cdef int n
    if(idx < 2 * (nside + 1) * nside):
        lower_b = (int)( (-1 + sqrt(1+2.0*idx)) / 2.0 )
        lower_idx = 2 * (lower_b + 1) * lower_b
        n = 4 * (lower_b + 1)

        l = lower_idx + ( idx - 1 - lower_idx + n) % n
        r = lower_idx + ( idx + 1 - lower_idx + n) % n

        dx = 2.0 * M_PI / (n)
        print(n)

    elif(idx >= 2*nside * (5 * nside - 1)):
        temp_idx = 12*nside*nside - idx - 1
        lower_b = (int)( (-1 + sqrt(1+2.0*temp_idx)) / 2.0 )
        n = 4 * (lower_b + 1)

        lower_idx = 12*nside*nside - 2 * (lower_b + 1) * lower_b - n

        
        l = lower_idx + ( idx - 1 - lower_idx + n) % n
        r = lower_idx + ( idx + 1 - lower_idx + n) % n

        dx = 2.0 * M_PI / (n)
        print([n, temp_idx, lower_b, lower_idx, ( idx + 1 - lower_idx + n)])

    else:
        lower_idx = (int)((idx - 2 * (nside + 1) * nside) / (4 * nside) ) * (4 * nside) \
            + 2 * (nside + 1) * nside 

        l = lower_idx + (idx - 1 - lower_idx + 4*nside) % (4*nside)
        r = lower_idx + (idx + 1 - lower_idx + 4*nside) % (4*nside)

        dx = 2.0 * M_PI / (4*nside)
        
    return [l, r, dx]


def getud(int idx, int u, int d, double dx, int nside):
    cdef int lower_b, lower_idx, temp_idx
    cdef int n
    if(idx < 2 * (nside + 1) * nside):
        lower_b = (int)( (-1 + sqrt(1+2.0*idx)) / 2.0 )
        lower_idx = 2 * (lower_b + 1) * lower_b
        n = 4 * (lower_b + 1)

        l = lower_idx + ( idx - 1 - lower_idx + n) % n
        r = lower_idx + ( idx + 1 - lower_idx + n) % n

        dx = 2.0 * M_PI / (n)
        print(n)

    elif(idx >= 2*nside * (5 * nside - 1)):
        temp_idx = 12*nside*nside - idx - 1
        lower_b = (int)( (-1 + sqrt(1+2.0*temp_idx)) / 2.0 )
        n = 4 * (lower_b + 1)

        lower_idx = 12*nside*nside - 2 * (lower_b + 1) * lower_b - n

        
        l = lower_idx + ( idx - 1 - lower_idx + n) % n
        r = lower_idx + ( idx + 1 - lower_idx + n) % n

        dx = 2.0 * M_PI / (n)
        print([n, temp_idx, lower_b, lower_idx, ( idx + 1 - lower_idx + n)])

    else:
        lower_idx = (int)((idx - 2 * (nside + 1) * nside) / (4 * nside) ) * (4 * nside) \
            + 2 * (nside + 1) * nside 

        l = lower_idx + (idx - 1 - lower_idx + 4*nside) % (4*nside)
        r = lower_idx + (idx + 1 - lower_idx + 4*nside) % (4*nside)

        dx = 2.0 * M_PI / (4*nside)
        
    return [l, r, dx]
