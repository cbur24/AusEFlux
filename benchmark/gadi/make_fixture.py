# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas", "rasterio"]
# ///
"""
Synthetic fixture for local end-to-end tests of the benchmark/gadi tools.

Writes, under OUTDIR (default ./fixture):
  members_ref/GPP_{lgbm,rf}_{1,2}.nc   4 reference members, 12 months 2025,
                                       24 x 30 px (lgbm: time/latitude/longitude,
                                       rf: time/y/x), NaN ocean corner and a few
                                       pixels with < 3 valid members
  members_new/GPP/...                  same members, one perturbed by 1e-6,
                                       one split into month parts parts/*_m{MM}.nc
  ref/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc   nearest-rank recombination of
                                       members_ref (lat descending, time units
                                       'days since 2003-01-15')
  new/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc   ref perturbed (~1 %), lat
                                       ascending, 2 extra longitude columns,
                                       a few mask flips, time 'days since 1970-01-01'
  parts/AusEFlux_GPP_500m_quantiles_2025_v2.1_m{MM}.nc   12 month parts of ref
  mvg.tif                              fake Major Vegetation Group raster in
                                       EPSG:3577 (Albers) at ~200 m
  features_{ref,new}/{NDVI,LST,ΔT}_500m.nc   36 months 2023-2025
  interim_{ref,new}/{NDVI,rain}/{var}_500m_{2024,2025}.nc  (new rain is
                                       vertically flipped -> orientation check)

    uv run make_fixture.py --out /path/to/fixture
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr

NY, NX, T = 24, 30, 12
DL = 0.005
LAT = (-30.0 - DL / 2 - DL * np.arange(NY)).round(4)          # descending
LON = (140.0 + DL / 2 + DL * np.arange(NX)).round(4)
TIMES = pd.date_range('2025-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)


def member_stack(rng, n=4):
    base = 50 + 40 * np.sin(np.arange(T) / 12 * 2 * np.pi)[:, None, None] \
        + 20 * rng.random((1, NY, NX))
    mem = np.stack([base + 10 * rng.standard_normal((T, NY, NX)) for _ in range(n)])
    mem = mem.astype('float32')
    mem[:, :, :4, :5] = np.nan                       # ocean corner
    mem[0:2, 3, 10, 12] = np.nan                     # 2 valid -> product NaN
    mem[0, :, 15, 20] = np.nan                       # 3 valid -> product finite
    return mem


def nearest(stack, q):
    vc = (~np.isnan(stack)).sum(0)
    s = np.sort(stack, axis=0)
    idx = np.clip(np.round(q * (vc - 1)).astype(np.int64), 0, stack.shape[0] - 1)
    v = np.take_along_axis(s, idx[None], 0)[0]
    return np.where(vc >= 3, v, np.nan).astype('float32')


def write_member(path, arr, name, dims):
    da = xr.DataArray(arr, dims=('time',) + dims,
                      coords={'time': TIMES, dims[0]: LAT, dims[1]: LON}, name=name)
    da.to_dataset().to_netcdf(path, encoding={
        'time': {'units': 'days since 2025-01-15', 'dtype': 'float64'}})


def product_ds(q25, q50, q75, flux='GPP', lat=LAT, lon=LON, times=TIMES):
    ds = xr.Dataset({f'{flux}_25th_percentile': (('time', 'latitude', 'longitude'), q25),
                     f'{flux}_median': (('time', 'latitude', 'longitude'), q50),
                     f'{flux}_75th_percentile': (('time', 'latitude', 'longitude'), q75)},
                    coords={'time': times, 'latitude': lat, 'longitude': lon})
    ds.attrs = {'short_name': flux, 'long_name': 'Gross Primary Productivity',
                'units': 'gC/m2/month', 'version': 'v2.1', 'crs': 'EPSG:4326'}
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='fixture', help='output directory')
    args = ap.parse_args()
    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    for d in ('members_ref', 'members_new/GPP/parts', 'ref', 'new', 'parts'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    rng = np.random.default_rng(0)

    # members
    mem = member_stack(rng)
    names = ['GPP_lgbm_1', 'GPP_lgbm_2', 'GPP_rf_1', 'GPP_rf_2']
    for m, nm in zip(mem, names):
        dims = ('latitude', 'longitude') if 'lgbm' in nm else ('y', 'x')
        write_member(os.path.join(out, 'members_ref', f'{nm}.nc'), m, 'GPP', dims)
    # new members: lgbm_1 identical, lgbm_2 perturbed, rf_1 as month parts
    for m, nm in zip(mem, names):
        a = m.copy()
        if nm == 'GPP_lgbm_2':
            a[5, 7, 9] += 1e-3
            a[2, 20, 25] = np.nan                # mask mismatch
        if nm == 'GPP_rf_1':
            for k in range(T):
                da = xr.DataArray(a[k:k + 1], dims=('time', 'latitude', 'longitude'),
                                  coords={'time': TIMES[k:k + 1], 'latitude': LAT,
                                          'longitude': LON}, name='GPP')
                da.to_dataset().to_netcdf(os.path.join(
                    out, 'members_new/GPP/parts', f'{nm}_m{k + 1:02d}.nc'))
            continue
        write_member(os.path.join(out, 'members_new/GPP', f'{nm}.nc'), a, 'GPP',
                     ('latitude', 'longitude'))

    # reference product = nearest-rank recombination
    q = [nearest(mem, p) for p in (0.25, 0.5, 0.75)]
    ref = product_ds(*q)
    enc = {v: {'zlib': True, 'chunksizes': (1, 8, 30)} for v in ref.data_vars}
    enc['time'] = {'units': 'days since 2003-01-15', 'dtype': 'float64'}
    ref.to_netcdf(os.path.join(out, 'ref/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc'),
                  encoding=enc)

    # month parts of ref (for merge_months)
    for k in range(T):
        ref.isel(time=[k]).to_netcdf(os.path.join(
            out, 'parts', f'AusEFlux_GPP_500m_quantiles_2025_v2.1_m{k + 1:02d}.nc'),
            encoding={'time': {'units': 'days since 1970-01-01'}})

    # new product: perturbed, flipped latitude, 2 extra lon columns, mask flips
    lon_new = np.concatenate([LON, (LON[-1] + DL * np.arange(1, 3)).round(4)])
    qs = []
    for a in q:
        b = a * (1 + 0.01 * rng.standard_normal(a.shape).astype('float32'))
        b = np.concatenate([b, np.full((T, NY, 2), 7.0, 'float32')], axis=2)
        b[:, 22, 3] = 42.0                       # ref NaN -> new valid
        b[1, 12, 12] = np.nan                    # ref valid -> new NaN
        qs.append(b[:, ::-1, :].copy())
    new = product_ds(*qs, lat=LAT[::-1].copy(), lon=lon_new)
    new.to_netcdf(os.path.join(out, 'new/AusEFlux_GPP_500m_quantiles_2025_v2.1.nc'),
                  encoding={'time': {'units': 'days since 1970-01-01'}})

    # fake MVG raster in Albers (EPSG:3577)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds('EPSG:4326', 'EPSG:3577', LON[0] - 0.02,
                                  LAT[-1] - 0.02, lon_new[-1] + 0.02, LAT[0] + 0.02)
    res = 200.0
    W, H = int((e - w) / res) + 1, int((n - s) / res) + 1
    cls = np.zeros((H, W), dtype='uint8')
    cls[:, : W // 3] = 5
    cls[:, W // 3: 2 * W // 3] = 19
    cls[:, 2 * W // 3:] = 22
    cls[: H // 4, :] = 3
    with rasterio.open(os.path.join(out, 'mvg.tif'), 'w', driver='GTiff', height=H,
                       width=W, count=1, dtype='uint8', crs='EPSG:3577',
                       transform=from_origin(w, n, res, res), nodata=0) as dst:
        dst.write(cls, 1)

    # features (36 months)
    ft = pd.date_range('2023-01-01', periods=36, freq='MS') + pd.Timedelta(days=14)
    for side in ('ref', 'new'):
        os.makedirs(os.path.join(out, f'features_{side}'), exist_ok=True)
    for var in ('NDVI', 'LST', 'ΔT'):
        base = rng.random((36, NY, NX)).astype('float32')
        base[:, :4, :5] = np.nan
        newv = base + 1e-3 * rng.standard_normal(base.shape).astype('float32')
        if var == 'ΔT':
            newv[15, 10, 10] = np.inf      # 2024-04
            newv[16, 11, 11] = np.nan     # 2024-05
        for side, arr in (('ref', base), ('new', newv)):
            dims = ('time', 'latitude', 'longitude') if side == 'new' else ('time', 'y', 'x')
            xr.DataArray(arr, dims=dims, coords={'time': ft, dims[1]: LAT, dims[2]: LON},
                         name=var).to_dataset().to_netcdf(
                os.path.join(out, f'features_{side}', f'{var}_500m.nc'))

    # interim (per-year files); new rain flipped vertically
    for var in ('NDVI', 'rain'):
        for year in (2024, 2025):
            tt = pd.date_range(f'{year}-01-01', periods=12, freq='MS') + pd.Timedelta(days=14)
            grad = np.linspace(0, 1, NY, dtype='float32')[None, :, None]
            base = (grad + 0.1 * rng.random((12, NY, NX))).astype('float32')
            newv = base[:, ::-1, :].copy() if var == 'rain' else base + 1e-4
            for side, arr in (('ref', base), ('new', newv)):
                d = os.path.join(out, f'interim_{side}', var)
                os.makedirs(d, exist_ok=True)
                xr.DataArray(arr, dims=('time', 'latitude', 'longitude'),
                             coords={'time': tt, 'latitude': LAT, 'longitude': LON},
                             name=var).to_dataset().to_netcdf(
                    os.path.join(d, f'{var}_500m_{year}.nc'))
    print(f'fixture written to {out}')


if __name__ == '__main__':
    main()
