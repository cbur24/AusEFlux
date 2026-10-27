"""Copy the smoke feature tree, injecting +-Inf into ~3 % of finite LAI_anom
and NDWI pixels (as LAI_anom has where its climatology is 0)."""
import os, sys, shutil
import numpy as np, xarray as xr
src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
rng = np.random.default_rng(3)
for v, sign in (('LAI_anom', 1), ('NDWI', -1)):
    p = f'{dst}/{v}_500m.nc'
    ds = xr.load_dataset(p); name = [k for k in ds.data_vars if ds[k].ndim == 3][0]
    a = ds[name].values.copy()
    sel = np.isfinite(a) & (rng.random(a.shape) < 0.03)
    a[sel] = sign * np.inf
    ds[name].values = a
    ds.to_netcdf(p + '.tmp'); os.replace(p + '.tmp', p)
    print(v, 'Inf injected:', int(sel.sum()))
