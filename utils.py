import numpy as npy
import healpy as hp

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

def interp(ta, dx, x_list):
    L = ta.shape * dx
    val = npy.zeros(len(x_list))
    for n in range(0, len(x_list)):
        x = x_list[n]
        #xl = (x / dx).astype(int)
        xl = npy.floor(x / dx).astype(int)
        
        for i in range(2):
            tempj = 0
            for j in range(2):
                tempk = 0
                for k in range(2):
                    tempk += ta[(xl[0] + i)%ta.shape[0], (xl[1]+j)%ta.shape[1], (xl[2] + k)%ta.shape[2]] \
                    *(1- abs((x[2] - (xl[2] + k) * dx[2] ) )/ dx[2])

                tempj += tempk * (1-abs((x[1] - (xl[1] + j) * dx[1] ) )/ dx[1])
            val[n] += tempj *(1 - abs((x[0] - (xl[0] + i) * dx[0] ) )/ dx[0])

    return val

# Given a periodic snapshot data, calculate its d/dr
# r maybe larger than the box size
# Only linearly interpolate a sphere (r, nise)
def f_r_derv(ta, dx, origin, r, nside):

    theta_list, phi_list = hp.pix2ang(nside, range(12*nside**2))
    x_list = npy.array([ \
                         r * npy.sin(theta_list) * npy.cos(phi_list) + origin[0],        \
                         r * npy.sin(theta_list) * npy.sin(phi_list) + origin[1],        \
                         r * npy.cos(theta_list) + origin[2]                             \
    ])

    x_list_T = npy.transpose(x_list)
    
    L = ta.shape * dx

    # calcuate derivatives on all mesh here 
    derv1 = fderv1(ta, dx[0], 0)
    derv2 = fderv1(ta, dx[1], 1)
    derv3 = fderv1(ta, dx[2], 2)

    return (x_list[0] * interp(derv1, dx, x_list_T) \
        + x_list[1] * interp(derv2, dx, x_list_T)  \
        + x_list[2] * interp(derv3, dx, x_list_T)  ) / r
    

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
