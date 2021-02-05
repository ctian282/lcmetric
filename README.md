# lc_metric
Metric reconstruction from light-cone data only

# Installation 
```shell
cd lc_metric
python setup.py build_ext --inplace
```
Then import 
```python
import lcmetric 
import lcmetric.lightcone as lightcone #main module building metric
import lcmetric.utils as ut #utilities written in python
import lcmetric.clightcone_CIC as lc_CIC #c utility that does CIC deposit to light-cone
import lcmetric.cutils as utC #c utility, under construction, for interpolation from cartesian mesh only
```

# Build metric
```python
lc = lightcone.Lightcone(NSIDE, epsilon = 1e-12,grid='healpy',  alm_iter = 50,
                    depth = 5, n_vcycles = 100, npre = 16, npost = 16, lmax = 2*NSIDE-1, verbose=False)
lc.init_from_slice(init_z, init_r, delta, vw, Phi_i, Pi_i, 
                       cosmo_paras, final_z, final_r, Phi_f)
lc.build_lcmetric()
```
See a sample jupyter nb for more detail
