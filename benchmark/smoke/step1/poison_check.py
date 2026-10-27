import sys, numpy as np, xarray as xr
ok = True
for eng in ('rasterio', 'odc'):
    for v in ('NDWI', 'kNDVI'):
        e = xr.open_dataarray(f'nan_{eng}_exclude/{v}/{v}_500m_2024.nc').values
        p = xr.open_dataarray(f'nan_{eng}_poison/{v}/{v}_500m_2024.nc').values
        ne, np_ = np.isnan(e), np.isnan(p)
        superset = not (ne & ~np_).any()
        strictly = np_.sum() > ne.sum()
        fin = ~np_
        same = np.array_equal(e[fin].view('u4'), p[fin].view('u4'))
        good = superset and strictly and same; ok &= good
        print(f'{"OK  " if good else "FAIL"} {eng:8s} {v:5s} NaN exclude={ne.sum()} poison={np_.sum()} '
              f'(+{(np_.sum()-ne.sum())/ne.size:.2%} of pixel-months) superset={superset} bit-identical where poison finite={same}')
for v in ('NDWI', 'kNDVI'):
    for m in ('exclude', 'poison'):
        a = xr.open_dataarray(f'nan_rasterio_{m}/{v}/{v}_500m_2024.nc').values
        b = xr.open_dataarray(f'nan_odc_{m}/{v}/{v}_500m_2024.nc').values
        both = np.isfinite(a) & np.isfinite(b)
        d = float(np.abs(a[both] - b[both]).max()); nq = np.array_equal(np.isnan(a), np.isnan(b))
        good = d <= 1e-5 and nq; ok &= good
        print(f'{"OK  " if good else "FAIL"} odc vs rasterio {v:5s} {m:7s} max|d|={d:.3g} nan_eq={nq}')
print('ALL OK' if ok else 'FAILURES')
