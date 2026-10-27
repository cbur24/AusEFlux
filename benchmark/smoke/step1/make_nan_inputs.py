"""Variant of data/input with MCD43A4 bands on a grid offset by half a pixel
(so each target pixel averages parts of 4 source pixels) and ~3 % of source
pixels NaN for the whole year (plus random NaN days)."""
import os, sys, shutil
import numpy as np, xarray as xr
src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
rng = np.random.default_rng(7)
mb = f'{dst}/MODIS/mosaic/MCD43A4.006'
hole = None
for f in sorted(os.listdir(mb)):
    p = f'{mb}/{f}'
    ds = xr.load_dataset(p); v = list(ds.data_vars)[0]
    ds = ds.assign_coords(latitude=ds.latitude - 0.0025, longitude=ds.longitude + 0.0025)
    a = ds[v].values.copy()
    if hole is None:
        hole = rng.random(a.shape[1:]) < 0.03
    a[:, hole] = np.nan
    a[rng.random(a.shape) < 0.05] = np.nan
    ds[v].values = a
    ds.to_netcdf(p + '.tmp'); os.replace(p + '.tmp', p)
print('NaN source pixels (all year):', int(hole.sum()), 'of', hole.size)
