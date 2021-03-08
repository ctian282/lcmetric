import numpy as npy
import scipy as scpy
import struct


# First direvative
def fderv1(ta, dx, dir):
    return (-0.5 * npy.roll(ta, 1, axis=dir) +
            0.5 * npy.roll(ta, -1, axis=dir)) / (dx)


def np_fderv1(ta, dx, dir):
    res = (-0.5 * npy.roll(ta, 1, axis=dir) +
           0.5 * npy.roll(ta, -1, axis=dir)) / (dx)
    res[0] = (npy.take(ta, 1, axis=dir) - npy.take(ta, 0, axis=dir)) / (dx)
    res[-1] = (npy.take(ta, -1, axis=dir) - npy.take(ta, -2, axis=dir)) / (dx)
    return res


def fderv2(ta, dx, dir):
    return (npy.roll(ta, 1, axis=dir) + npy.roll(ta, -1, axis=dir) -
            2.0 * ta) / (dx**2)


def np_fderv2(ta, dx, dir):
    res = (npy.roll(ta, 1, axis=dir) + npy.roll(ta, -1, axis=dir) -
           2.0 * ta) / (dx**2)
    res[0] = (npy.take(ta, 2, axis = dir) + npy.take(ta, 0, axis = dir) \
              - 2 * npy.take(ta, 1, axis = dir)) / (dx)**2
    res[-1] = (npy.take(ta, -3, axis = dir) + npy.take(ta, -1, axis = dir) \
              - 2 * npy.take(ta, -2, axis = dir)) / (dx)**2
    return res


def flap(ta, dx):
    return fderv2(ta, dx[0], 0) + fderv2(ta, dx[1], 1) + fderv2(ta, dx[2], 2)


def interp_lc(ta, r, r_offset, dr):
    rl = npy.floor((r - r_offset) / dr).astype(int)
    if (rl + 1 >= ta.shape[0]):
        raise ValueError('r is too large!')
    elif (rl < 0):
        raise ValueError('r is too small!')
    ans = (1 - npy.abs(r - rl * dr) / dr) * ta[rl] \
        + (npy.abs(r - rl * dr) / dr) * ta[rl + 1]
    return ans


def interp(ta, dx, x_list, grid='healpy'):

    if (grid == 'healpy'):
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
                        tempk += ta[(xl[0] + i) % ta.shape[0],
                                    (xl[1]+j) % ta.shape[1],
                                    (xl[2] + k) % ta.shape[2]]\
                                    *(1- abs((x[2] - (xl[2] + k) * dx[2] ) )/ dx[2])

                    tempj += tempk * (1 - abs(
                        (x[1] - (xl[1] + j) * dx[1])) / dx[1])
                val[n] += tempj * (1 - abs(
                    (x[0] - (xl[0] + i) * dx[0])) / dx[0])

        return val
    else:
        print("Waning, unsupported grid type.")
    return 0


# Given a periodic snapshot data, calculate its d/dr
# r maybe larger than the box size
# Only linearly interpolate a sphere (r, nise)
def f_r_derv(ta, dx, r, x_list, grid='healpy'):

    # calcuate derivatives on all mesh here
    derv1 = fderv1(ta, dx[0], 0)
    derv2 = fderv1(ta, dx[1], 1)
    derv3 = fderv1(ta, dx[2], 2)

    if (grid == 'healpy'):
        return (x_list[:,0] * interp(derv1, dx, x_list, grid) \
                + x_list[:,1] * interp(derv2, dx, x_list, grid)  \
                + x_list[:,2] * interp(derv3, dx, x_list, grid)  ) / r
    elif (grid == 'pysh'):
        return (x_list[:,:,0] * interp(derv1, dx, x_list, grid) \
                + x_list[:,:,1] * interp(derv2, dx, x_list, grid)  \
                + x_list[:,:,2] * interp(derv3, dx, x_list, grid)  ) / r


def inverse_derv(field, L, N, dir):

    ks = 2.0 * npy.pi * npy.fft.fftfreq(N, L / N)
    ks[0] = 1

    field_fft = npy.fft.fftn(field)

    if (dir == 0):
        field_fft /= 1.0j * ks[:, None, None]
    elif (dir == 1):
        field_fft /= 1.0j * ks[None, :, None]
    else:
        field_fft /= 1.0j * ks[None, None, :]

    field_fft[0, 0, 0] = 0

    return npy.real(npy.fft.ifftn(field_fft))


def inverse_Lap(field, L, N):
    k2s = fftFreqs(L, N)**2
    k2s[0, 0, 0] = 1

    field = scale(field, 1.0 / k2s**2)
    field = -field
    return field


def scale(field, pks):
    field_fft = npy.fft.fftn(field)
    field_fft *= npy.sqrt(pks)
    field_fft[0, 0, 0] = 0
    return npy.real(npy.fft.ifftn(field_fft))


def fftFreqs(L, N):
    ks = 2.0 * npy.pi * npy.fft.fftfreq(N, L / N)
    return npy.sqrt(ks[:, None, None]**2 + ks[None, :, None]**2 +
                    ks[None, None, :]**2)


# 1/H(z)
def Hint(z, Hubble, Omega_m, Omega_L):
    return 1 / (Hubble * npy.sqrt(Omega_m * (1 + z)**3 + Omega_L))


# H(z)
def H(z, Hubble, Omega_m, Omega_L):
    return (Hubble / (1 + z) * npy.sqrt(Omega_m * (1 + z)**3 + Omega_L))


# integrate 1/H(z) - r, to inversely solve z for given r
def inverse_Hint(z, r, Hubble, Omega_m, Omega_L):
    return npy.abs(
        scpy.integrate.quad(Hint, 0, z, args=(Hubble, Omega_m, Omega_L))[0] -
        r)


# Convert comoving r to z
def r2z(r, Hubble, Omega_m, Omega_L):
    return scpy.optimize.minimize_scalar(inverse_Hint,
                                         args=(r, Hubble, Omega_m, Omega_L)).x


def unf_read_file(file, p_list=[], np=7):
    with open(file, mode="rb") as f:
        tot_n = 0
        cnt = 0
        while (True):
            cnt += 1
            r = f.read(4)
            if not r: break

            a1 = struct.unpack('i', r)

            r = f.read(a1[0])
            n = struct.unpack('i', r)

            r = f.read(8)
            a, b = struct.unpack('2i', r)

            r = f.read(b)
            p_list.extend(struct.unpack(str(n[0] * np) + 'f', r))

            r = f.read(4)
            tot_n += n[0]
    f.close()
    return tot_n
