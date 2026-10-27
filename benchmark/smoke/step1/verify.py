import sys, glob, os
import numpy as np, xarray as xr, netCDF4 as nc4
out = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else None
expected = {v: 2024 for v in 'NDWI kNDVI NDVI LAI LST VegH Tmin Tmax kTavg Tavg SRAD rain VPD'.split()}
ok = True
for v, y in list(expected.items()) + [('rain', 2023)]:
    p = f'{out}/{v}/{v}_500m_{y}.nc'
    if not os.path.exists(p): print('MISSING', p); ok = False; continue
    da = xr.open_dataarray(p)
    t = da.time.to_index()
    chk = (da.dims == ('time', 'latitude', 'longitude') and da.shape == (12, 50, 60) and da.dtype == np.float32
           and all(t.day == 15) and list(t.month) == list(range(1, 13)) and all(t.year == y)
           and da.latitude.values[0] > da.latitude.values[-1])
    with nc4.Dataset(p) as d: ch = d[v].chunking()
    fin = float(np.isfinite(da.values).mean())
    diff = ''
    if ref:
        r = xr.open_dataarray(f'{ref}/{v}/{v}_500m_{y}.nc').values
        a = da.values; both = np.isfinite(a) & np.isfinite(r)
        diff = f' maxabs_vs_ref={np.nanmax(np.abs(a[both]-r[both])) if both.any() else np.nan:.3g} nan_eq={np.array_equal(np.isnan(a), np.isnan(r))}'
    print(f'{"OK " if chk else "BAD"} {v}_{y} shape={da.shape} dtype={da.dtype} chunks={ch} finite={fin:.3f}{diff}')
    ok &= chk and ch == [1, 50, 60]
print('ALL OK' if ok else 'FAILURES')
