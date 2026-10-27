import os, sys, numpy as np, pandas as pd, xarray as xr
out = sys.argv[1]; NLAT, NLON = 600, 800
LAT = (-20.0 - 0.0025 - 0.005*np.arange(NLAT)).round(4); LON = (130.0 + 0.0025 + 0.005*np.arange(NLON)).round(4)
for var in ['NDWI', 'SRAD']:
    rng = np.random.default_rng(len(var))
    d = os.path.join(out, 'interim', var); os.makedirs(d, exist_ok=True)
    for y in range(2003, 2015):
        t = pd.DatetimeIndex([pd.Timestamp(f'{y}-{m:02d}-15') for m in range(1, 13)])
        a = (0.3 + 0.1*rng.standard_normal((12, NLAT, NLON))).astype('float32'); a[rng.random(a.shape) < 0.05] = np.nan
        xr.DataArray(a, dims=('time','latitude','longitude'), coords={'time': t, 'latitude': LAT, 'longitude': LON}, name=var).to_netcdf(os.path.join(d, f'{var}_500m_{y}.nc'))
