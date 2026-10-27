"""Wide-grid odc vs rasterio check: 50 x 8400 source at 0.005 deg with float32
coords (OzWALD-like), warped onto (a) the same grid, (b) a grid offset by 1/3
pixel. Compares the module given in argv[1] (path to step1_harmonisation.py)."""
import sys, importlib.util
import numpy as np, xarray as xr
from rasterio.enums import Resampling
spec = importlib.util.spec_from_file_location('s1', sys.argv[1]); s1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s1)
n_lat, n_lon, res = 50, 8400, 0.005
lat = (-10.0025 - res * np.arange(n_lat)).astype('float32')       # descending, float32
lon = (112.0025 + res * np.arange(n_lon)).astype('float32')
L, O = np.meshgrid(lat.astype('f8'), lon.astype('f8'), indexing='ij')
arr = (np.sin(O * 40.0) * 10 + L).astype('float32')               # strong E-W structure
def template(lat_t, lon_t):
    return xr.DataArray(np.ones((len(lat_t), len(lon_t)), 'u1'), dims=('latitude', 'longitude'),
                        coords={'latitude': lat_t, 'longitude': lon_t})
cases = {'same grid': template(lat, lon),
         'offset 1/3 px': template((lat.astype('f8') - res / 3).astype('float32'), (lon.astype('f8') + res / 3).astype('float32'))}
ok = True
for name, tpl in cases.items():
    for rs in (Resampling.bilinear, Resampling.average, Resampling.nearest):
        out = {}
        for eng in ('rasterio', 'odc'):
            s1.RUN['engine'] = eng
            out[eng] = s1._make_warper(lat, lon, tpl, rs)(arr)
        a, b = out['rasterio'], out['odc']
        if name == 'same grid':
            ident = float(np.abs(a - arr).max()); ok &= ident == 0; print(f'     same-grid identity max|rio-src|={ident:.3g}')
        both = np.isfinite(a) & np.isfinite(b)
        d = float(np.abs(a[both] - b[both]).max())
        nan_a, nan_b = int(np.isnan(a).sum()), int(np.isnan(b).sum())
        # column shift of odc vs rasterio in the east (best-matching lag)
        east = slice(n_lon - 400, n_lon - 10)
        lags = {k: np.nanmean(np.abs(b[5:-5, east] - np.roll(a, k, axis=1)[5:-5, east])) for k in range(-6, 7)}
        lag = min(lags, key=lags.get)
        good = d <= 1e-5 and nan_a == nan_b and lag == 0
        ok &= good
        print(f'{"OK  " if good else "FAIL"} {name:14s} {rs.name:8s} max|odc-rio|={d:.3g} NaN rio={nan_a} odc={nan_b} east lag={lag}')
print('ALL OK' if ok else 'FAILURES')
