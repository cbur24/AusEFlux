"""Variant smoke feature trees (argv: src dst mode).
win        rows 10:25 of every feature + global attribute auseflux_lat_rows='10:25'
win_noattr rows 10:25 without the attribute (must be refused: urban mask mismatch)
tbad       feature B with its 2024 time stamps moved to the 1st of the month"""
import os, sys, shutil, glob
import numpy as np, pandas as pd, xarray as xr
src, dst, mode = sys.argv[1:4]
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
for p in sorted(glob.glob(f'{src}/*_500m.nc')):
    q = f'{dst}/{os.path.basename(p)}'
    ds = xr.load_dataset(p)
    if mode.startswith('win'):
        ds = ds.isel(latitude=slice(10, 25))
        if mode == 'win':
            ds.attrs['auseflux_lat_rows'] = '10:25'
    elif mode == 'tbad' and os.path.basename(p) == 'B_500m.nc':
        t = pd.DatetimeIndex(ds.time.values)
        ds = ds.assign_coords(time=[x.replace(day=1) if x.year == 2024 else x for x in t])
    ds.to_netcdf(q)
print(mode, 'features ->', dst)
