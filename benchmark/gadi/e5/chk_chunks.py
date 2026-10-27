import xarray as xr, numpy as np, glob
fs = sorted(glob.glob('../smoke_step2/data/interim/NDVI/*.nc'))
ds = xr.open_mfdataset(fs, chunks={'latitude': (3, 25, 12)})
print(xr.__version__, ds[list(ds.data_vars)[0]].chunks)
