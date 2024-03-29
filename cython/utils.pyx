#Cython version of utils.py
#Currently, the only difference is the interperlation function, which is
#rewritten by c

import numpy as npy
import healpy as hp

cimport cutils as cutils
cimport numpy as npy
cimport cython

ctypedef cython.floating real_t
# Using geometric unit,
# when L = 3e5 Mpc, value of H = h *100
L_UNIT = 3e5 # Mpc
c = 3e5 # km/s

def fderv1(ta, dx, dir):
    return (-0.5 * npy.roll(ta,1, axis = dir) + 0.5 * npy.roll(ta,-1, axis = dir))/(dx)

def fderv2(ta, dx, dir):
    return ( npy.roll(ta,1, axis = dir) + npy.roll(ta,-1, axis = dir) - 2.0 * ta)/(dx**2)

def flap(ta, dx):
    return fderv2(ta, dx[0], 0) + fderv2(ta, dx[1], 1) + fderv2(ta, dx[2], 2)


cpdef interp(real_t[:,:,::1] ta, double [::1] dx, real_t[:,::1] x_list):

    cdef long long ns = x_list.shape[0]

    if real_t is float:
        dtype = npy.float32
    else:
        dtype = npy.float64

    cdef npy.ndarray[real_t, ndim=1, mode='c'] val = npy.zeros(ns, dtype=dtype)

    cutils._interp(&ta[0,0,0], &dx[0], ta.shape[0], ta.shape[1], ta.shape[2],\
                   &x_list[0,0], ns, &val[0])
    return val

# Given a periodic snapshot data, calculate its d/dr
# r maybe larger than the box size
# Only linearly interpolate a sphere (r, nise)
def f_r_derv(real_t[:,:, ::1]ta, double [::1] dx,
             double r, real_t[:, ::1] x_list):

    # calcuate derivatives on all mesh here
    cdef real_t[:, :, ::1] derv1 = \
        npy.ascontiguousarray(fderv1(ta, dx[0], 0))
    cdef real_t[:, :, ::1] derv2 = \
        npy.ascontiguousarray(fderv1(ta, dx[1], 1))
    cdef real_t[:, :, ::1] derv3 = \
        npy.ascontiguousarray(fderv1(ta, dx[2], 2))

    return (  (x_list[:,0]) * interp(derv1, dx, x_list)
            + (x_list[:,1]) * interp(derv2, dx, x_list)
            + (x_list[:,2]) * interp(derv3, dx, x_list)) / r

def inverse_derv(field, L, N, dir):

    ks = 2.0*npy.pi*npy.fft.fftfreq(N, L/N)
    ks[0] = 1

    field_fft = npy.fft.fftn(field)


    if(dir == 0):
        field_fft /= 1.0j*ks[:,None,None]
    elif(dir == 1):
        field_fft /= 1.0j*ks[None,:,None]
    else:
        field_fft /= 1.0j*ks[None,None,:]

    field_fft[0,0,0] = 0

    return npy.real(npy.fft.ifftn(field_fft))


def inverse_Lap(field, L, N):
    k2s = fftFreqs(L, N)**2
    k2s[0,0,0] = 1

    field = scale(field, 1.0/k2s**2)
    field = -field
    return field

def scale(field, pks) :
    field_fft = npy.fft.fftn(field)
    field_fft *= npy.sqrt( pks )
    field_fft[0,0,0] = 0
    return npy.real(npy.fft.ifftn(field_fft))

def fftFreqs(L, N) :
    ks = 2.0*npy.pi*npy.fft.fftfreq(N, L/N)
    return npy.sqrt( ks[:,None,None]**2 + ks[None,:,None]**2 + ks[None,None,:]**2 )
