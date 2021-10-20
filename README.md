# LC-Metric
Metric reconstruction from light cone data only

# Prerequisite

`Healpy`, `Healpix-Cxx`, `cython`


# Installation
```shell
cd lcmetric
python setup.py build_ext --inplace
```
Then import
```python

import lcmetric
import lcmetric.lightcone as lc # main module building metric
import lcmetric.utils as # utilities written in python
import lcmetric.cgeodesic as geo # geodesic module doing ray-tracing
import lcmetric.lensing as lensing # lensing module
import lcmetric.clcdensity as dens # module restoring light cone from snapshots

# Build metric
```python

# See lightcone.py for details for parameter setting
lc = lc.LightconeFromCone(lc_path, init_snap_path, final_snap_path, origin,
                         cosmo_paras, L_snap, N_snap, init_z, final_z,
                         NR, NSIDE, zel_z, zel_snap_path, cone_type='UNFORMATTED',
                                snap_type='Gadget1', lensing_kappa=True, dtype='f4',
                                chunk=400000000, linear_Omega=False, depo_method='CIC', smoothing=True)
# Multigrid relaxing
lc_relx.init_build_lcmetric()
```
