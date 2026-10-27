"""Copy the smoke interim; add a `spatial_ref` coordinate (like the reference
odc interim files) to every file of years <= 2004 (E5 hybrid-interim case)."""
import os, shutil, sys, glob
import numpy as np, xarray as xr
S = sys.argv[1]
src, dst = f'{S}/data/interim', f'{S}/data_mixed/interim'
shutil.rmtree(f'{S}/data_mixed', ignore_errors=True)
shutil.copytree(src, dst)
n = 0
for f in sorted(glob.glob(f'{dst}/*/*.nc')):
    yr = int(f[-7:-3])
    if yr > 2004:
        continue
    ds = xr.load_dataset(f)
    ds = ds.assign_coords(spatial_ref=np.int32(0))
    for v in ds.data_vars:
        ds[v].encoding['coordinates'] = 'spatial_ref'
    os.remove(f); ds.to_netcdf(f); n += 1
print('added spatial_ref to', n, 'files')
