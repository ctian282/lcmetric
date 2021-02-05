# lc_metric
Metric reconstruction from light-cone data only

#Installation 
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

