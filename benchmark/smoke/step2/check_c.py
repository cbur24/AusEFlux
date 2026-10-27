import sys, numpy as np, xarray as xr
S = sys.argv[1]
ok = True
for var in ['NDWI','SRAD','Tavg','VPD','kNDVI','LAI','rain','rain_cml3','rain_cml6','rain_cml12']:
    n = xr.open_dataarray(f'{S}/out_none/{var}_anom_500m.nc')
    d = xr.open_dataarray(f'{S}/out_def/{var}_anom_500m.nc').values
    src = xr.open_dataarray(f'{S}/out_def/{var}_500m.nc').sel(time=n.time).values
    clim = xr.open_dataset(f'{S}/out_def/climatology/{var}_clim_500m.nc')['clim'].values
    exp = (src / clim[n.time.dt.month.values - 1]).astype('float32')
    nv = n.values
    same_none = np.array_equal(exp.view('u4'), nv.view('u4')) or np.array_equal(exp, nv, equal_nan=True)
    exp1 = np.where(np.isfinite(src) & ~np.isfinite(exp), np.float32(1), exp)
    same_one = np.array_equal(exp1, d, equal_nan=True)
    zero = (clim[n.time.dt.month.values - 1] == 0) & np.isfinite(src)
    print(f'  {var:10s} clim==0&src finite: {zero.sum():4d}  none: nan {np.isnan(nv[zero]).sum()} inf {np.isinf(nv[zero]).sum()} | '
          f'none==src/clim {same_none}  one==guarded {same_one}  one has no nonfinite where src finite: {bool(np.isfinite(d[np.isfinite(src)]).all())}')
    ok &= same_none and same_one
tz = (xr.open_dataset(f'{S}/out_def/climatology/Tavg_clim_500m.nc')['clim'].values == 0).sum()
rz = (xr.open_dataset(f'{S}/out_def/climatology/rain_clim_500m.nc')['clim'].values == 0).sum()
print('  clim==0 pixels: Tavg', tz, 'rain', rz)
sys.exit(0 if ok and tz and rz else 1)
