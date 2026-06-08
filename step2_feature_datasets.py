"""
Step 2: Feature Dataset Construction for AusEFlux.

Combines per-year interim files from Step 1 into multi-year feature datasets,
applies gap-filling, computes vegetation fractions (Donohue et al. 2009),
cumulative rainfall, fractional anomalies, ΔT, and C4 grass fraction.

Memory strategy: data is processed in spatial tiles (latitude bands) with the
full time series loaded for each band.  Peak RSS stays under ~4 GB even for
20+ years of data on the 500 m grid.

Usage:
    uv run step2_feature_datasets.py
    uv run step2_feature_datasets.py --interim_path interim_500m/ --results_path features_500m/
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "rioxarray>=0.19.0",
#     "xarray>=2025.1.0",
#     "netCDF4",
#     "rasterio",
#     "dask",
# ]
# ///

import argparse
import gc
import os
import sys
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')

import netCDF4 as nc4
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS as _CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as _from_bounds
from rasterio.warp import reproject as _rasterio_reproject

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_PATH = '/home/prl900/Downloads/wald_eflux'
DEFAULT_INTERIM = os.path.join(BASE_PATH, 'interim_500m')
DEFAULT_RESULTS = os.path.join(BASE_PATH, 'features_500m')
DEFAULT_STATIC  = os.path.join(BASE_PATH, 'input', 'static')
TARGET_GRID = '500m'

EXCLUDE_VARS = {'Tmin', 'Tmax', 'kTavg'}
INTERP_GAPFILL_VARS = {'NDWI', 'kNDVI'}

CLIM_START = '2003'
CLIM_END   = '2022'

ANOM_VARS = ['NDWI', 'SRAD', 'Tavg', 'VPD', 'kNDVI', 'LAI',
             'rain', 'rain_cml3', 'rain_cml6', 'rain_cml12']

MAX_TILE_MEM = 2.0e9  # ~2 GB per tile array

os.environ.setdefault('GDAL_CACHEMAX', '256')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tile_rows(n_times, n_lon, max_mem=MAX_TILE_MEM):
    """Number of latitude rows per tile to stay under *max_mem* bytes."""
    bytes_per_row = n_times * n_lon * 4
    rows = int(max_mem / bytes_per_row) if bytes_per_row > 0 else 400
    return min(max(rows, 50), 400)


def _clim_subset(da):
    """
    Select the climatology reference period (2003–2022) from *da*.
    Falls back to all available data if the reference period has no overlap.
    """
    subset = da.sel(time=slice(CLIM_START, CLIM_END))
    if subset.sizes['time'] == 0:
        return da
    return subset


def _feature_vars(interim_path):
    """List variable directories, excluding intermediates."""
    return sorted(d for d in os.listdir(interim_path)
                  if os.path.isdir(os.path.join(interim_path, d))
                  and d not in EXCLUDE_VARS)


def _nc_files(directory):
    """Sorted .nc file paths in *directory*."""
    return sorted(os.path.join(directory, f)
                  for f in os.listdir(directory) if f.endswith('.nc'))


def _data_var(ds):
    """Return the first non-coordinate data variable name."""
    skip = {'crs', 'spatial_ref'}
    return next(v for v in ds.data_vars if v not in skip and ds[v].ndim > 0)


def _coords(da):
    """Return (lat, lon) float32 arrays rounded to 4 d.p."""
    lat = np.array([round(float(v), 4) for v in da.latitude.values], dtype='float32')
    lon = np.array([round(float(v), 4) for v in da.longitude.values], dtype='float32')
    return lat, lon


# ---------------------------------------------------------------------------
# Spatial helpers (shared with step 1)
# ---------------------------------------------------------------------------

def _affine(lat, lon):
    dlon = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.005
    dlat = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.005
    west  = float(min(lon)) - dlon / 2
    east  = float(max(lon)) + dlon / 2
    south = float(min(lat)) - dlat / 2
    north = float(max(lat)) + dlat / 2
    return _from_bounds(west, south, east, north, len(lon), len(lat))


def _make_warper(src_lat, src_lon, dst_lat, dst_lon, resampling):
    """Pre-compute rasterio reprojection callable: warp(arr_2d) → arr_2d."""
    crs = _CRS.from_epsg(4326)
    src_tr = _affine(src_lat, src_lon)
    dst_tr = _affine(dst_lat, dst_lon)
    out_shape = (len(dst_lat), len(dst_lon))

    def warp(arr):
        src = np.ascontiguousarray(arr.astype('float32', copy=False))
        dst = np.full(out_shape, np.float32(np.nan), dtype='float32')
        _rasterio_reproject(
            source=src, destination=dst,
            src_transform=src_tr, src_crs=crs,
            dst_transform=dst_tr, dst_crs=crs,
            resampling=resampling,
            src_nodata=np.float32(np.nan),
            dst_nodata=np.float32(np.nan),
        )
        return dst

    return warp


# ---------------------------------------------------------------------------
# NetCDF4 output: pre-allocated (time, lat, lon) file for tile writes
# ---------------------------------------------------------------------------

@contextmanager
def _nc_full(path, var_name, lat, lon, times):
    """
    Create an output NetCDF and yield the open Dataset.

    Caller writes spatial tiles via ``nc[var][:, start:end, :] = data``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds = nc4.Dataset(path, 'w', format='NETCDF4')
    try:
        ds.createDimension('time', len(times))
        ds.createDimension('latitude', len(lat))
        ds.createDimension('longitude', len(lon))

        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units    = 'days since 1970-01-01'
        tv.calendar = 'proleptic_gregorian'
        epoch = pd.Timestamp('1970-01-01')
        for i, t in enumerate(times):
            tv[i] = float((pd.Timestamp(t) - epoch).days)

        lv = ds.createVariable('latitude', 'f4', ('latitude',))
        lv.units = 'degrees_north'
        lv[:] = lat

        xv = ds.createVariable('longitude', 'f4', ('longitude',))
        xv.units = 'degrees_east'
        xv[:] = lon

        ds.createVariable(
            var_name, 'f4', ('time', 'latitude', 'longitude'),
            fill_value=np.float32(np.nan), zlib=True, complevel=4,
            chunksizes=(1, min(200, len(lat)), min(200, len(lon))),
        )
        yield ds
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Step 2a: Combine per-year files + gap-fill
# ---------------------------------------------------------------------------

def _combine_gapfill(interim_path, results_path, verbose=False):
    """
    For each variable, combine annual interim files into a single multi-year
    feature file with gap-filling.  Rain includes the prior year and is written
    as ``rain_…_extrayear.nc``; a trimmed copy is made afterwards.
    """
    vars_to_process = _feature_vars(interim_path)

    # Determine the first non-rain year (for rain trimming later)
    ref_var = next((v for v in vars_to_process if v != 'rain'), vars_to_process[0])
    ref_files = _nc_files(os.path.join(interim_path, ref_var))
    with xr.open_mfdataset(ref_files) as ref_ds:
        first_year = str(pd.DatetimeIndex(ref_ds.time.values).year.min())

    for var in vars_to_process:
        out_name = (f'rain_{TARGET_GRID}_extrayear.nc' if var == 'rain'
                    else f'{var}_{TARGET_GRID}.nc')
        out_path = os.path.join(results_path, out_name)

        if os.path.exists(out_path):
            if verbose:
                print(f'  {var}: exists, skipping')
            continue

        var_dir = os.path.join(interim_path, var)
        files = _nc_files(var_dir)
        if not files:
            print(f'  WARNING: no files for {var}, skipping')
            continue

        if verbose:
            print(f'  {var}')

        ds = xr.open_mfdataset(files, chunks={})
        dvar = _data_var(ds)
        da = ds[dvar]

        n_time = da.sizes['time']
        n_lat  = da.sizes['latitude']
        n_lon  = da.sizes['longitude']
        tile_r = _tile_rows(n_time, n_lon)
        lat, lon = _coords(da)
        times = da.time.values

        with _nc_full(out_path, dvar, lat, lon, times) as nc_out:
            for lat_s in range(0, n_lat, tile_r):
                lat_e = min(lat_s + tile_r, n_lat)
                tile = da.isel(latitude=slice(lat_s, lat_e)).compute(
                    scheduler='synchronous')

                if var in INTERP_GAPFILL_VARS:
                    clim = _clim_subset(tile) \
                               .groupby('time.month').mean()
                    anom = tile.groupby('time.month') - clim
                    anom = anom.interpolate_na(
                        dim='time', method='linear', limit=2)
                    filled = anom.groupby('time.month') + clim
                    filled = filled.drop_vars('month')
                    filled = filled.groupby('time.month').fillna(clim)
                    filled = filled.drop_vars('month')
                else:
                    clim = _clim_subset(tile) \
                               .groupby('time.month').mean()
                    filled = tile.groupby('time.month').fillna(clim)
                    filled = filled.drop_vars('month')

                nc_out[dvar][:, lat_s:lat_e, :] = \
                    filled.values.astype('float32')
                del tile, filled
                gc.collect()

        ds.close()
        gc.collect()

    # Trimmed rain (excluding the prior year)
    extrayear_path = os.path.join(results_path,
                                  f'rain_{TARGET_GRID}_extrayear.nc')
    rain_path = os.path.join(results_path, f'rain_{TARGET_GRID}.nc')

    if os.path.exists(extrayear_path) and not os.path.exists(rain_path):
        if verbose:
            print('  Trimming rain to target period')

        ds = xr.open_dataset(extrayear_path, chunks={})
        dvar = _data_var(ds)
        da = ds[dvar].sel(time=slice(first_year, '2052'))

        n_time = da.sizes['time']
        n_lat  = da.sizes['latitude']
        n_lon  = da.sizes['longitude']
        tile_r = _tile_rows(n_time, n_lon)
        lat, lon = _coords(da)
        times = da.time.values

        with _nc_full(rain_path, dvar, lat, lon, times) as nc_out:
            for lat_s in range(0, n_lat, tile_r):
                lat_e = min(lat_s + tile_r, n_lat)
                tile = da.isel(latitude=slice(lat_s, lat_e)).compute(
                    scheduler='synchronous')
                nc_out[dvar][:, lat_s:lat_e, :] = \
                    tile.values.astype('float32')
                del tile

        ds.close()


# ---------------------------------------------------------------------------
# Step 2b: Vegetation fractions (Donohue et al. 2009)
# ---------------------------------------------------------------------------

def _vegetation_fractions(results_path, static_path, verbose=False,
                          ndvi_max=0.91):
    """
    Decompose gap-filled NDVI into trees (persistent), grass (recurrent),
    and bare fractions.  Annual maxima are broadcast to monthly time steps.
    """
    ndvi_path    = os.path.join(results_path, f'NDVI_{TARGET_GRID}.nc')
    ndvi_min_path = os.path.join(static_path,
                                 f'ndvi_of_baresoil_{TARGET_GRID}.nc')

    for p, label in [(ndvi_path, 'Gap-filled NDVI'),
                     (ndvi_min_path, 'Bare-soil NDVI map')]:
        if not os.path.exists(p):
            print(f'  WARNING: {label} not found ({p}), skipping veg fractions')
            return

    out_paths = {v: os.path.join(results_path, f'{v}_{TARGET_GRID}.nc')
                 for v in ('trees', 'grass', 'bare')}
    if all(os.path.exists(p) for p in out_paths.values()):
        if verbose:
            print('  trees/grass/bare: exist, skipping')
        return

    if verbose:
        print('  Computing vegetation fractions')

    ndvi_ds  = xr.open_dataset(ndvi_path, chunks={})
    dvar     = _data_var(ndvi_ds)
    ndvi_da  = ndvi_ds[dvar]

    ndvi_min_ds = xr.open_dataset(ndvi_min_path)
    ndvi_min_var = _data_var(ndvi_min_ds)
    ndvi_min_2d = ndvi_min_ds[ndvi_min_var].values.astype('float32')

    n_time = ndvi_da.sizes['time']
    n_lat  = ndvi_da.sizes['latitude']
    n_lon  = ndvi_da.sizes['longitude']
    times  = ndvi_da.time.values
    lat, lon = _coords(ndvi_da)

    years = sorted(set(pd.DatetimeIndex(times).year))
    monthly_times = [pd.Timestamp(f'{y}-{m:02d}-15')
                     for y in years for m in range(1, 13)]

    # Smaller tiles: veg fractions create several intermediate arrays
    tile_r = _tile_rows(n_time, n_lon, max_mem=1.2e9)

    nc_files = {}
    try:
        # Create output files
        for vname in ('trees', 'grass', 'bare'):
            p = out_paths[vname]
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            nf = nc4.Dataset(p, 'w', format='NETCDF4')
            nf.createDimension('time', len(monthly_times))
            nf.createDimension('latitude', len(lat))
            nf.createDimension('longitude', len(lon))

            tv = nf.createVariable('time', 'f8', ('time',))
            tv.units = 'days since 1970-01-01'
            tv.calendar = 'proleptic_gregorian'
            epoch = pd.Timestamp('1970-01-01')
            for i, t in enumerate(monthly_times):
                tv[i] = float((t - epoch).days)

            lv = nf.createVariable('latitude', 'f4', ('latitude',))
            lv.units = 'degrees_north'
            lv[:] = lat

            xv = nf.createVariable('longitude', 'f4', ('longitude',))
            xv.units = 'degrees_east'
            xv[:] = lon

            nf.createVariable(
                vname, 'f4', ('time', 'latitude', 'longitude'),
                fill_value=np.float32(np.nan), zlib=True, complevel=4,
                chunksizes=(1, min(200, len(lat)), min(200, len(lon))),
            )
            nc_files[vname] = nf

        # Process in spatial tiles
        for lat_s in range(0, n_lat, tile_r):
            lat_e = min(lat_s + tile_r, n_lat)
            if verbose and lat_s % (tile_r * 5) == 0:
                print(f'    rows {lat_s}–{lat_e} / {n_lat}')

            ndvi_tile = ndvi_da.isel(latitude=slice(lat_s, lat_e)).compute(
                scheduler='synchronous')
            ndvi_min_tile = ndvi_min_2d[lat_s:lat_e, :]

            # Fractional total vegetation cover
            ft = (ndvi_tile - ndvi_min_tile) / (ndvi_max - ndvi_min_tile)
            ft = ft.clip(0, 1)

            # Persistent (trees): rolling min then mean (Donohue 2009 eq 1–2)
            persist = ft.rolling(time=7, min_periods=1).min()
            persist = persist.rolling(time=9, min_periods=1).mean()

            # Recurrent (grass): eq 3
            recurrent = ft - persist

            # Correction: eq 4 & 5
            persist = xr.where(recurrent < 0,
                               persist - np.abs(recurrent), persist)
            recurrent = ft - persist

            persist   = persist.clip(min=0)
            recurrent = recurrent.clip(min=0)
            del ft, ndvi_tile

            # Annual max
            persist_ann   = persist.resample(time='YE').max()
            recurrent_ann = recurrent.resample(time='YE').max()
            bare_ann      = (1 - (persist_ann + recurrent_ann)).clip(min=0)
            del persist, recurrent

            # Broadcast annual → monthly (constant within each year)
            tile_lat_len = lat_e - lat_s
            trees_arr = np.empty((len(monthly_times), tile_lat_len, n_lon),
                                 dtype='float32')
            grass_arr = np.empty_like(trees_arr)
            bare_arr  = np.empty_like(trees_arr)

            out_idx = 0
            for y in years:
                ys = str(y)
                p_slice = persist_ann.sel(time=ys).values.squeeze()
                r_slice = recurrent_ann.sel(time=ys).values.squeeze()
                b_slice = bare_ann.sel(time=ys).values.squeeze()
                for _ in range(12):
                    trees_arr[out_idx] = p_slice
                    grass_arr[out_idx] = r_slice
                    bare_arr[out_idx]  = b_slice
                    out_idx += 1

            nc_files['trees']['trees'][:, lat_s:lat_e, :] = trees_arr
            nc_files['grass']['grass'][:, lat_s:lat_e, :] = grass_arr
            nc_files['bare']['bare'][:, lat_s:lat_e, :]   = bare_arr

            del trees_arr, grass_arr, bare_arr
            del persist_ann, recurrent_ann, bare_ann
            gc.collect()

    finally:
        for nf in nc_files.values():
            nf.close()

    ndvi_ds.close()
    ndvi_min_ds.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2c: Cumulative rainfall
# ---------------------------------------------------------------------------

def _cumulative_rainfall(results_path, verbose=False):
    """
    Compute 3-, 6-, and 12-month rolling cumulative rainfall from the
    extrayear rain file.  Remove the extrayear file when done.

    Loads each spatial tile eagerly and computes all three rolling windows
    on the in-memory array (avoids dask graph overhead on tiny chunks).
    """
    extrayear_path = os.path.join(results_path,
                                  f'rain_{TARGET_GRID}_extrayear.nc')
    if not os.path.exists(extrayear_path):
        print('  WARNING: rain extrayear file not found, skipping')
        return

    windows = [(3, 'rain_cml3'), (6, 'rain_cml6'), (12, 'rain_cml12')]

    # Check which windows still need computing
    windows_todo = []
    for window, out_var in windows:
        out_path = os.path.join(results_path, f'{out_var}_{TARGET_GRID}.nc')
        if os.path.exists(out_path):
            if verbose:
                print(f'  {out_var}: exists, skipping')
        else:
            windows_todo.append((window, out_var))

    if not windows_todo:
        # Still need to clean up extrayear
        if verbose:
            print('  Removing rain extrayear file')
        os.remove(extrayear_path)
        return

    ds = xr.open_dataset(extrayear_path, chunks={})
    dvar = _data_var(ds)
    da = ds[dvar]

    n_time_full = da.sizes['time']
    n_lat = da.sizes['latitude']
    n_lon = da.sizes['longitude']
    lat, lon = _coords(da)

    # Determine output time range (trimmed to CLIM_START+)
    all_times = pd.DatetimeIndex(da.time.values)
    time_mask = all_times >= pd.Timestamp(f'{CLIM_START}-01-01')
    out_time_indices = np.where(time_mask)[0]
    out_times = all_times[time_mask].values
    tile_r = _tile_rows(n_time_full, n_lon)

    if verbose:
        for _, out_var in windows_todo:
            print(f'  {out_var}')

    # Open all output files
    nc_outs = {}
    try:
        for window, out_var in windows_todo:
            out_path = os.path.join(results_path,
                                    f'{out_var}_{TARGET_GRID}.nc')
            nc_outs[out_var] = {}
            nc_outs[out_var]['window'] = window
            # Create output file via context manager, but manage lifecycle here
            os.makedirs(os.path.dirname(os.path.abspath(out_path)),
                        exist_ok=True)
            nf = nc4.Dataset(out_path, 'w', format='NETCDF4')
            nf.createDimension('time', len(out_times))
            nf.createDimension('latitude', len(lat))
            nf.createDimension('longitude', len(lon))

            tv = nf.createVariable('time', 'f8', ('time',))
            tv.units = 'days since 1970-01-01'
            tv.calendar = 'proleptic_gregorian'
            epoch = pd.Timestamp('1970-01-01')
            for i, t in enumerate(out_times):
                tv[i] = float((pd.Timestamp(t) - epoch).days)

            lv = nf.createVariable('latitude', 'f4', ('latitude',))
            lv.units = 'degrees_north'
            lv[:] = lat

            xv = nf.createVariable('longitude', 'f4', ('longitude',))
            xv.units = 'degrees_east'
            xv[:] = lon

            nf.createVariable(
                out_var, 'f4', ('time', 'latitude', 'longitude'),
                fill_value=np.float32(np.nan), zlib=True, complevel=4,
                chunksizes=(1, min(200, len(lat)), min(200, len(lon))),
            )
            nc_outs[out_var]['nc'] = nf

        # Process in spatial tiles: load eagerly, compute all windows at once
        for lat_s in range(0, n_lat, tile_r):
            lat_e = min(lat_s + tile_r, n_lat)

            # Load tile eagerly — all time steps, this lat band
            tile = da.isel(latitude=slice(lat_s, lat_e)).compute(
                scheduler='synchronous')

            for window, out_var in windows_todo:
                # Rolling sum on in-memory xarray (no dask overhead)
                cml = tile.rolling(time=window, min_periods=window).sum()
                # Select output time range
                cml_out = cml.isel(time=out_time_indices).values \
                              .astype('float32')
                nc_outs[out_var]['nc'][out_var][:, lat_s:lat_e, :] = cml_out
                del cml, cml_out

            del tile
            gc.collect()

    finally:
        for info in nc_outs.values():
            info['nc'].close()

    ds.close()

    # Remove extrayear file now that cumulative products are written
    if verbose:
        print('  Removing rain extrayear file')
    os.remove(extrayear_path)
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2d: ΔT = LST − Tavg
# ---------------------------------------------------------------------------

def _delta_t(results_path, verbose=False):
    """Compute land surface temperature minus air temperature, per month."""
    lst_path  = os.path.join(results_path, f'LST_{TARGET_GRID}.nc')
    tavg_path = os.path.join(results_path, f'Tavg_{TARGET_GRID}.nc')
    out_path  = os.path.join(results_path, f'ΔT_{TARGET_GRID}.nc')

    if os.path.exists(out_path):
        if verbose:
            print('  ΔT: exists, skipping')
        return

    for p, label in [(lst_path, 'LST'), (tavg_path, 'Tavg')]:
        if not os.path.exists(p):
            print(f'  WARNING: {label} not found ({p}), skipping ΔT')
            return

    if verbose:
        print('  ΔT')

    lst_ds  = xr.open_dataset(lst_path)
    tavg_ds = xr.open_dataset(tavg_path)
    lst_var  = _data_var(lst_ds)
    tavg_var = _data_var(tavg_ds)

    lat, lon = _coords(lst_ds[lst_var])
    times = lst_ds[lst_var].time.values
    n_time = len(times)

    # Process one time step at a time (each slice ~228 MB)
    with _nc_full(out_path, 'ΔT', lat, lon, times) as nc_out:
        for t in range(n_time):
            lst_s  = lst_ds[lst_var].isel(time=t).values.astype('float32')
            tavg_s = tavg_ds[tavg_var].isel(time=t).values.astype('float32')
            nc_out['ΔT'][t, :, :] = lst_s - tavg_s
            del lst_s, tavg_s

    lst_ds.close()
    tavg_ds.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2e: Fractional anomalies
# ---------------------------------------------------------------------------

def _fractional_anomalies(results_path, verbose=False):
    """
    Compute fractional anomalies: variable / monthly_climatology(2003–2022).
    """
    for var in ANOM_VARS:
        in_path  = os.path.join(results_path, f'{var}_{TARGET_GRID}.nc')
        out_path = os.path.join(results_path,
                                f'{var}_anom_{TARGET_GRID}.nc')

        if os.path.exists(out_path):
            if verbose:
                print(f'  {var}_anom: exists, skipping')
            continue

        if not os.path.exists(in_path):
            print(f'  WARNING: {var} not found ({in_path}), '
                  f'skipping {var}_anom')
            continue

        if verbose:
            print(f'  {var}_anom')

        ds = xr.open_dataset(in_path, chunks={})
        dvar = _data_var(ds)
        da = ds[dvar]

        n_time = da.sizes['time']
        n_lat  = da.sizes['latitude']
        n_lon  = da.sizes['longitude']
        tile_r = _tile_rows(n_time, n_lon)
        lat, lon = _coords(da)

        # Output times trimmed to 2003+
        da_trimmed = da.sel(time=slice(CLIM_START, '2052'))
        out_times = da_trimmed.time.values

        anom_var = f'{var}_anom'

        with _nc_full(out_path, anom_var, lat, lon, out_times) as nc_out:
            for lat_s in range(0, n_lat, tile_r):
                lat_e = min(lat_s + tile_r, n_lat)
                tile = da.isel(latitude=slice(lat_s, lat_e)).compute(
                    scheduler='synchronous')

                clim = _clim_subset(tile) \
                           .groupby('time.month').mean('time')
                anom = tile.groupby('time.month') / clim
                anom = anom.drop_vars('month') \
                           .sel(time=slice(CLIM_START, '2052'))

                # Guard the fractional anomaly against 0/0 (NaN) and x/0 (Inf),
                # which occur in e.g. zero-rainfall months where the 2003-2022
                # climatology is also zero.  Where the source value is finite
                # but the anomaly is not, set it to 1.0 (== at climatological
                # level) instead of leaving NaN/Inf.  Otherwise step3's no-data
                # mask drops these (often productive) pixels, shrinking spatial
                # coverage relative to the reference product.
                src = tile.sel(time=slice(CLIM_START, '2052'))
                anom = anom.where(
                    ~(np.isfinite(src) & ~np.isfinite(anom)), 1.0)

                nc_out[anom_var][:, lat_s:lat_e, :] = \
                    anom.values.astype('float32')
                del tile, clim, anom
                gc.collect()

        ds.close()
        gc.collect()


# ---------------------------------------------------------------------------
# Step 2f: C4 grass fraction
# ---------------------------------------------------------------------------

def _c4_grass_fraction(results_path, static_path, verbose=False):
    """
    Combine a static C4 grass coverage map with the dynamic grass fraction
    to produce a time-varying C4 grass fraction.
    """
    c4_path = os.path.join(static_path, 'Aust_C4_grass_cover_percentage.tif')
    grass_path = os.path.join(results_path, f'grass_{TARGET_GRID}.nc')
    mask_path  = os.path.join(static_path, f'land_sea_mask_{TARGET_GRID}.nc')
    out_path   = os.path.join(results_path, f'C4_grass_{TARGET_GRID}.nc')

    if os.path.exists(out_path):
        if verbose:
            print('  C4_grass: exists, skipping')
        return

    for p, label in [(c4_path, 'C4 grass TIF'),
                     (grass_path, 'Grass fraction'),
                     (mask_path, 'Land mask')]:
        if not os.path.exists(p):
            print(f'  WARNING: {label} not found ({p}), skipping C4 grass')
            return

    if verbose:
        print('  C4_grass')

    # Load and reproject C4 coverage to target grid
    grass_ds = xr.open_dataset(grass_path)
    grass_var = _data_var(grass_ds)
    dst_lat, dst_lon = _coords(grass_ds[grass_var])

    mask_ds = xr.open_dataset(mask_path)
    mask_var = _data_var(mask_ds)
    mask_arr = mask_ds[mask_var].values

    with rasterio.open(c4_path) as src:
        c4_data = src.read(1).astype('float32')
        transform = src.transform
        h, w = c4_data.shape
        # Build source coordinate arrays from the rasterio transform
        src_lon = np.array([transform.c + (col + 0.5) * transform.a
                            for col in range(w)], dtype='float32')
        src_lat = np.array([transform.f + (row + 0.5) * transform.e
                            for row in range(h)], dtype='float32')

    # Mask invalid values and reproject
    c4_data[c4_data < 0] = np.float32(np.nan)
    warp = _make_warper(src_lat, src_lon, dst_lat, dst_lon, Resampling.average)
    c4_reprojected = warp(c4_data)
    del c4_data

    # Apply land mask and convert percentage → fraction
    c4_reprojected = np.where(mask_arr, c4_reprojected, np.float32(np.nan))
    c4_reprojected /= 100.0

    # Write C4_grass = grass × C4_fraction, one time step at a time
    times = grass_ds[grass_var].time.values
    n_time = len(times)

    with _nc_full(out_path, 'C4_grass', dst_lat, dst_lon, times) as nc_out:
        for t in range(n_time):
            grass_slice = grass_ds[grass_var].isel(time=t).values \
                          .astype('float32')
            nc_out['C4_grass'][t, :, :] = grass_slice * c4_reprojected
            del grass_slice

    grass_ds.close()
    mask_ds.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def create_feature_datasets(interim_path=None, results_path=None,
                            static_path=None, verbose=False):
    """
    Run the full Step 2 pipeline:
      2a  Combine + gap-fill
      2b  Vegetation fractions
      2c  Cumulative rainfall
      2d  ΔT
      2e  Fractional anomalies
      2f  C4 grass fraction
    """
    if interim_path is None:
        interim_path = DEFAULT_INTERIM
    if results_path is None:
        results_path = DEFAULT_RESULTS
    if static_path is None:
        static_path = DEFAULT_STATIC

    os.makedirs(results_path, exist_ok=True)

    if verbose:
        print('Step 2a: Combining datasets + gap-filling')
    _combine_gapfill(interim_path, results_path, verbose=verbose)

    if verbose:
        print('Step 2b: Vegetation fractions')
    _vegetation_fractions(results_path, static_path, verbose=verbose)

    if verbose:
        print('Step 2c: Cumulative rainfall')
    _cumulative_rainfall(results_path, verbose=verbose)

    if verbose:
        print('Step 2d: LST minus Tair (ΔT)')
    _delta_t(results_path, verbose=verbose)

    if verbose:
        print('Step 2e: Fractional anomalies')
    _fractional_anomalies(results_path, verbose=verbose)

    if verbose:
        print('Step 2f: C4 grass fraction')
    _c4_grass_fraction(results_path, static_path, verbose=verbose)

    if verbose:
        print('Step 2 complete.')
        print(f'Outputs written to: {results_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 2: Feature dataset construction.')
    parser.add_argument('--interim_path', type=str, default=None,
                        help=f'Step 1 output directory. '
                             f'Default: {DEFAULT_INTERIM}')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Output directory. Default: {DEFAULT_RESULTS}')
    parser.add_argument('--static_path', type=str, default=None,
                        help=f'Static input files. '
                             f'Default: {DEFAULT_STATIC}')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    create_feature_datasets(
        interim_path=args.interim_path,
        results_path=args.results_path,
        static_path=args.static_path,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
