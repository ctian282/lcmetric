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

def interp(ta, dx, x_list, grid = 'healpy'):

    if(grid == 'healpy'):
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
    elif(grid == 'pysh'):
        val = npy.zeros((x_list.shape[0], x_list.shape[1]))
        for m in range(0, x_list.shape[0]):
            for n in range(0, x_list.shape[1]):
                x = x_list[m,n]
                xl = npy.floor(x / dx).astype(int)
            
                for i in range(2):
                    tempj = 0
                    for j in range(2):
                        tempk = 0
                        for k in range(2):
                            tempk += ta[(xl[0] + i)%ta.shape[0], (xl[1]+j)%ta.shape[1], (xl[2] + k)%ta.shape[2]] \
                                *(1- abs((x[2] - (xl[2] + k) * dx[2] ) )/ dx[2])
                            
                        tempj += tempk * (1-abs((x[1] - (xl[1] + j) * dx[1] ) )/ dx[1])
                    val[m,n] += tempj *(1 - abs((x[0] - (xl[0] + i) * dx[0] ) )/ dx[0])
        return val
    else:
        Print("Waning, unsupported grid type.")
        return 0



# Given a periodic snapshot data, calculate its d/dr
# r maybe larger than the box size
# Only linearly interpolate a sphere (r, nise)
def f_r_derv(ta, dx, origin, r, x_list, grid= 'healpy'):


    # calcuate derivatives on all mesh here 
    derv1 = fderv1(ta, dx[0], 0)
    derv2 = fderv1(ta, dx[1], 1)
    derv3 = fderv1(ta, dx[2], 2)

    if(grid == 'healpy'):
        return (x_list[:,0] * interp(derv1, dx, x_list, grid) \
                + x_list[:,1] * interp(derv2, dx, x_list, grid)  \
                + x_list[:,2] * interp(derv3, dx, x_list, grid)  ) / r
    elif(grid == 'pysh'):
        return (x_list[:,:,0] * interp(derv1, dx, x_list, grid) \
                + x_list[:,:,1] * interp(derv2, dx, x_list, grid)  \
                + x_list[:,:,2] * interp(derv3, dx, x_list, grid)  ) / r
        
    

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
