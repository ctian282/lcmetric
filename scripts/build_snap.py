import dill as dl
import numpy as npy
import gc
import sys

args = sys.argv[1:]
sys.path.append('../../')

import lcmetric
import lcmetric.lightcone as lc
import lcmetric.utils as ut
import lcmetric.cgeodesic as geo
import lcmetric.lensing as lensing
import lcmetric.clcdensity as dens


def dl_dump(path, name, c):
    with open(path + name, 'wb') as f:
        dl.dump(c, f)
    f.close()


def dl_load(path, name):
    with open(path + name, 'rb') as f:
        temp = dl.load(f)
    f.close()
    return temp


n_threads = int(args[7])

# Setting initial cosmic parameters
cosmo_paras = {'h': 0.69, 'Omega_m': 0.31, 'Omega_L': 0.69}
init_zel_z = 9  #Redshift where n-body simulation starts
init_z = 0.30014
init_a = 1 / (1 + init_z)
final_z = 0.03
final_a = 1 / (1 + final_z)
zel_z = 9
zel_a = 1 / (1 + zel_z)

L_unit = 3e5

N_snap = 1024  #snapshot mesh resolution
L_snap = 1024 / cosmo_paras['h'] / L_unit

origin = npy.array([0, 0, 0.0])
NR = 1024
NSIDE = 512

# Working directory path
path = args[1]
# All snaps path
snaps_path = args[1] + args[2]
# Initial snap path
init_snap_path = args[1] + args[3]
# Final snap path
final_snap_path = args[1] + args[4]
# Zel snap path
zel_snap_path = args[1] + args[5]
# lc data path
lc_path = args[1] + args[6]

print(path)
print(snaps_path)
print(init_snap_path)
print(final_snap_path)
print(zel_snap_path)
print(lc_path)
print(n_threads)

lc_snaps_16 = lc.LightconeFromSnaps(snaps_path,
                                    n_threads,
                                    origin,
                                    cosmo_paras,
                                    L_snap,
                                    N_snap,
                                    -2,
                                    0,
                                    NR,
                                    NSIDE,
                                    zel_z,
                                    zel_snap_path,
                                    snap_type='Gadget1',
                                    NR_is_N_snap=True,
                                    need_reduce=True)
dl_dump(path, 'lc_snaps_16', lc_snaps_16)
del lc_snaps_16

gc.collect()

lc_snaps_64 = lc.LightconeFromSnaps(snaps_path,
                                    n_threads,
                                    origin,
                                    cosmo_paras,
                                    L_snap,
                                    N_snap,
                                    -2,
                                    1,
                                    NR,
                                    NSIDE,
                                    zel_z,
                                    zel_snap_path,
                                    snap_type='Gadget1',
                                    NR_is_N_snap=True)
dl_dump(path, 'lc_snaps_64', lc_snaps_64)
del lc_snaps_64
