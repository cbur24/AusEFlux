# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "xarray>=2025.1.0", "netCDF4", "rasterio"]
# ///
"""Synthetic interim + static tree for the step2 smoke test.

usage: make_fake.py OUTDIR [--first_year 2003] [--rain_first_year 2002] [--last_year 2006]
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin

VARS = ['kNDVI', 'kTavg', 'LAI', 'LST', 'NDVI', 'NDWI', 'rain', 'SRAD',
        'Tavg', 'Tmax', 'Tmin', 'VegH', 'VPD']
NLAT, NLON = (int(x) for x in __import__('os').environ.get('FAKE_GRID', '40x50').split('x'))
RES = 0.005
LAT = (-20.0 - RES / 2 - RES * np.arange(NLAT)).round(4)
LON = (130.0 + RES / 2 + RES * np.arange(NLON)).round(4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('--first_year', type=int, default=2003)
    p.add_argument('--rain_first_year', type=int, default=2002)
    p.add_argument('--last_year', type=int, default=2006)
    a = p.parse_args()
    interim = os.path.join(a.out, 'interim')
    static = os.path.join(a.out, 'static')
    os.makedirs(static, exist_ok=True)

    for var in VARS:
        y0 = a.rain_first_year if var == 'rain' else a.first_year
        years = list(range(y0, a.last_year + 1))
        times = pd.DatetimeIndex([pd.Timestamp(f'{y}-{m:02d}-15')
                                  for y in years for m in range(1, 13)])
        rng = np.random.default_rng(sum(map(ord, var)) * 7919)
        nt = len(times)
        month = times.month.values[:, None, None]
        seas = np.sin(2 * np.pi * month / 12.0)
        if var == 'NDVI':
            # Strongly varying NDVI so annual max(trees)+max(grass) > 1 somewhere
            arr = rng.uniform(0.05, 0.95, (nt, NLAT, NLON))
        elif var in ('kNDVI', 'NDWI'):
            arr = 0.3 + 0.2 * seas + 0.05 * rng.standard_normal((nt, NLAT, NLON))
        elif var == 'rain':
            arr = np.maximum(0, 50 + 40 * seas + 20 * rng.standard_normal((nt, NLAT, NLON)))
            # block with zero rain every January -> clim==0 -> 0/0
            jan = times.month.values == 1
            arr[jan, :5, :5] = 0.0
        elif var == 'Tavg':
            arr = 20 + 8 * seas + rng.standard_normal((nt, NLAT, NLON))
            # block whose January values are +1/-1 alternating -> clim exactly 0
            for i, t in enumerate(times):
                if t.month == 1:
                    arr[i, -5:, -5:] = 1.0 if (t.year % 2) else -1.0
        else:
            arr = 10 + 5 * seas + rng.standard_normal((nt, NLAT, NLON))
        arr = arr.astype('float32')
        # random NaN gaps
        arr[rng.random(arr.shape) < 0.05] = np.nan
        if var in ('kNDVI', 'NDWI'):
            # long gaps (4 months) in some pixels to exercise interpolate_na limit
            arr[10:14, 5:10, 5:10] = np.nan
            arr[:2, 20:25, 20:25] = np.nan   # leading gap
        d = os.path.join(interim, var)
        os.makedirs(d, exist_ok=True)
        for y in years:
            sel = times.year == y
            da = xr.DataArray(arr[sel], dims=('time', 'latitude', 'longitude'),
                              coords={'time': times[sel], 'latitude': LAT,
                                      'longitude': LON}, name=var)
            da.to_netcdf(os.path.join(d, f'{var}_500m_{y}.nc'))

    rng = np.random.default_rng(1)
    coords = {'latitude': LAT, 'longitude': LON}
    xr.DataArray(rng.uniform(0.05, 0.15, (NLAT, NLON)).astype('float32'),
                 dims=('latitude', 'longitude'), coords=coords, name='NDVI') \
        .to_netcdf(os.path.join(static, 'ndvi_of_baresoil_500m.nc'))
    lsm = np.ones((NLAT, NLON), 'int8'); lsm[:, :3] = 0
    xr.DataArray(lsm, dims=('latitude', 'longitude'), coords=coords,
                 name='landsea_mask_500m') \
        .to_netcdf(os.path.join(static, 'land_sea_mask_500m.nc'))
    uwm = np.zeros((NLAT, NLON), 'int8'); uwm[30:33, 30:33] = 1
    xr.DataArray(uwm, dims=('latitude', 'longitude'), coords=coords,
                 name='urban_water_mask_500m') \
        .to_netcdf(os.path.join(static, 'urban_water_mask_500m.nc'))
    # C4 tif at 0.01 deg covering the grid
    c4 = rng.uniform(0, 100, (NLAT // 2, NLON // 2)).astype('float32')
    c4[0, 0] = -9999
    tr = from_origin(130.0, -20.0, 0.01, 0.01)
    with rasterio.open(os.path.join(static, 'Aust_C4_grass_cover_percentage.tif'), 'w',
                       driver='GTiff', height=c4.shape[0], width=c4.shape[1], count=1,
                       dtype='float32', crs='EPSG:4326', transform=tr) as dst:
        dst.write(c4, 1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
