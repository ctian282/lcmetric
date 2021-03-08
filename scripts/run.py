import sys
sys.path.append("/home/chris/Dropbox/Document/Metric reconstruction/")

import lcmetric
import lcmetric.scripts.snap_to_phi
cosmo_paras = {'h': 0.69, 'Omega_m': 0.31, 'Omega_L': 0.69}

L_snap = 512
N_snap = 256
steps = '256_'
lcmetric.scripts.snap_to_phi.snap_to_phi(
    '/home/chris/Desktop/code_test/gadget/cone_test_256_z0p300.[0-9]*',
    cosmo_paras, L_snap, N_snap, steps)
