"""
Step 1: Spatiotemporal Harmonisation of AusEFlux input datasets.

Reads raw satellite and climate data, reprojects everything to the 500 m
target grid, resamples to monthly time steps (time-stamped at the 15th of
each month), and writes one NetCDF file per variable to the interim output
directory.

Each variable is skipped if its output file already exists, so the script
is safe to re-run after a partial failure.

Memory & I/O strategy
---------------------
All source files are opened with ``xr.open_dataset(path, chunks={})`` so
that xarray uses dask with the file's **native on-disk chunk layout**.
Monthly aggregation is then expressed as a lazy dask graph and computed
with ``scheduler='synchronous'``, which processes one compressed block at
a time — reading each block from disk exactly once instead of once per
time-step (the naive approach causes > 1 TB of redundant I/O for the
daily 500 m datasets with 25-timestep time-chunks).

Each reprojected 2-D monthly slice is written directly to the output
NetCDF file so the full 3-D time series is never held in RAM.

Reprojection uses rasterio directly (not rioxarray per-slice) via a
pre-computed affine transform, avoiding repeated xarray/GDAL overhead.

Usage:
    uv run step1_harmonisation.py --year 2024
    python step1_harmonisation.py --year 2024
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
import rioxarray  # registers the .rio accessor
import xarray as xr
from rasterio.crs import CRS as _CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as _from_bounds
from rasterio.warp import reproject as _rasterio_reproject


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_GRID = '500m'

BASE_PATH = '/home/prl900/Downloads/wald_eflux/input'
LOCAL_RESULTS = '/home/prl900/Downloads/wald_eflux/interim_500m/'
STATIC_FILES = {
    'land_mask': f'{BASE_PATH}/static/land_sea_mask_500m.nc',
    'VegH':      f'{BASE_PATH}/static/VegH_1km_2007_2010.nc',
}

# Cap GDAL's internal block cache so it cannot silently consume several GB
os.environ.setdefault('GDAL_CACHEMAX', '256')   # MB


def input_files(year):
    y = str(year)
    yp = str(year - 1)
    b = BASE_PATH
    files = {
        'MCD43A4 Band 1 (Red)':  f'{b}/MODIS/mosaic/MCD43A4.006/MCD43A4.006.b01.500m_0620_0670nm_nbar.{y}.nc',
        'MCD43A4 Band 2 (NIR)':  f'{b}/MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.{y}.nc',
        'MCD43A4 Band 5 (SWIR)': f'{b}/MODIS/mosaic/MCD43A4.006/MCD43A4.006.b05.500m_1230_1250nm_nbar.{y}.nc',
        'MYD11A1 LST':           f'{b}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.{y}.nc',
        'OzWALD NDVI':           f'{b}/OzWALD/8day/NDVI/OzWALD.NDVI.{y}.nc',
        'OzWALD LAI':            f'{b}/OzWALD/8day/LAI/OzWALD.LAI.{y}.nc',
        'OzWALD Tmin':           f'{b}/OzWALD/daily/meteo/Tmin/OzWALD.Tmin.{y}.nc',
        'OzWALD Tmax':           f'{b}/OzWALD/daily/meteo/Tmax/OzWALD.Tmax.{y}.nc',
        'OzWALD kTavg':          f'{b}/OzWALD/daily/meteo/kTavg/OzWALD.kTavg.{y}.nc',
        'SILO radiation':        f'{b}/SILO/radiation/{y}.radiation.nc',
        'SILO daily rain':       f'{b}/SILO/daily_rain/{y}.daily_rain.nc',
        'SILO vapour pressure':  f'{b}/SILO/vp/{y}.vp.nc',
        f'SILO daily rain {yp}': f'{b}/SILO/daily_rain/{yp}.daily_rain.nc',
    }
    if year < 2022:
        files['MYD11A1 QC'] = f'{b}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.{y}.nc'
    return files


def check_inputs(year):
    all_files = {**STATIC_FILES, **input_files(year)}
    missing = [f'{label}: {path}' for label, path in all_files.items()
               if not os.path.exists(path)]
    if missing:
        print('ERROR: The following required input files are missing:')
        for m in missing:
            print(f'  {m}')
        return False
    return True


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _spatial_dim_names(da):
    """Return (lat_name, lon_name) regardless of naming convention."""
    for lat in ('latitude', 'lat', 'y'):
        if lat in da.dims:
            lat_name = lat
            break
    else:
        raise ValueError(f'No lat dimension in {list(da.dims)}')
    for lon in ('longitude', 'lon', 'x'):
        if lon in da.dims:
            lon_name = lon
            break
    else:
        raise ValueError(f'No lon dimension in {list(da.dims)}')
    return lat_name, lon_name


def _affine(lat, lon):
    """
    Compute a rasterio Affine transform for a regular lat/lon grid.
    Works for both ascending and descending latitude arrays.
    """
    dlon = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.005
    dlat = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.005
    west  = float(min(lon)) - dlon / 2
    east  = float(max(lon)) + dlon / 2
    south = float(min(lat)) - dlat / 2
    north = float(max(lat)) + dlat / 2
    return _from_bounds(west, south, east, north, len(lon), len(lat))


def _make_warper(src_lat, src_lon, template, resampling):
    """
    Pre-compute source and destination affine transforms once, return a
    lightweight callable::

        warp(arr_2d_float32) -> arr_2d_float32

    Using rasterio directly avoids creating an xarray DataArray per call
    and prevents GDAL object accumulation inside the reprojection loop.
    """
    crs = _CRS.from_epsg(4326)
    src_tr = _affine(src_lat, src_lon)

    # _affine() always builds a north-up, west-first transform (row 0 = north,
    # col 0 = west).  Sources whose latitude is ascending (row 0 = south, e.g.
    # SILO) or whose longitude is descending must be flipped to match, otherwise
    # rasterio reprojects them upside-down / mirrored.  MODIS and OzWALD are
    # already north-up so are left untouched.
    src_flip_lat = len(src_lat) > 1 and src_lat[0] < src_lat[-1]
    src_flip_lon = len(src_lon) > 1 and src_lon[0] > src_lon[-1]

    lat_d, lon_d = _spatial_dim_names(template)
    dst_lat = template.coords[lat_d].values
    dst_lon = template.coords[lon_d].values
    dst_tr  = _affine(dst_lat, dst_lon)
    out_shape = (len(dst_lat), len(dst_lon))

    def warp(arr):
        # Orient the source north-up / west-first to match src_tr.
        if src_flip_lat:
            arr = arr[::-1, :]
        if src_flip_lon:
            arr = arr[:, ::-1]
        # Ensure (lat, lon) row-major layout expected by rasterio
        src = np.ascontiguousarray(arr.astype('float32', copy=False))
        dst = np.full(out_shape, np.float32(np.nan), dtype='float32')
        _rasterio_reproject(
            source=src,
            destination=dst,
            src_transform=src_tr,
            src_crs=crs,
            dst_transform=dst_tr,
            dst_crs=crs,
            resampling=resampling,
            src_nodata=np.float32(np.nan),
            dst_nodata=np.float32(np.nan),
        )
        return dst

    return warp


def _template_coords(template):
    """Return (lat, lon) float32 arrays from the template, rounded to 4 d.p."""
    lat_d, lon_d = _spatial_dim_names(template)
    lat = np.array([round(float(v), 4) for v in template.coords[lat_d].values],
                   dtype='float32')
    lon = np.array([round(float(v), 4) for v in template.coords[lon_d].values],
                   dtype='float32')
    return lat, lon


# ---------------------------------------------------------------------------
# NetCDF4 streaming writer
# ---------------------------------------------------------------------------

@contextmanager
def _nc_writer(path, var_name, lat, lon):
    """
    Context manager: open a netCDF4 file for streaming monthly writes.
    Yields the open Dataset; each month is written via _write_slice().
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds = nc4.Dataset(path, 'w', format='NETCDF4')
    try:
        ds.createDimension('time', None)
        ds.createDimension('latitude', len(lat))
        ds.createDimension('longitude', len(lon))

        tv = ds.createVariable('time', 'f8', ('time',))
        tv.units    = 'days since 1970-01-01'
        tv.calendar = 'proleptic_gregorian'

        lv = ds.createVariable('latitude', 'f4', ('latitude',))
        lv.units = 'degrees_north'
        lv[:] = lat

        xv = ds.createVariable('longitude', 'f4', ('longitude',))
        xv.units = 'degrees_east'
        xv[:] = lon

        ds.createVariable(var_name, 'f4', ('time', 'latitude', 'longitude'),
                          fill_value=np.float32(np.nan), zlib=True, complevel=4)
        yield ds
    finally:
        ds.close()


def _write_slice(nc_ds, var_name, t_idx, timestamp, data_2d):
    """Write one 2-D float32 slice to the open streaming netCDF4 dataset."""
    epoch = pd.Timestamp('1970-01-01')
    nc_ds['time'][t_idx] = float((timestamp - epoch).days)
    nc_ds[var_name][t_idx] = data_2d.astype('float32', copy=False)


# ---------------------------------------------------------------------------
# Dask-aware monthly aggregation
# ---------------------------------------------------------------------------

def _monthly_means_gen(da, year, agg='mean'):
    """
    Generator: yield (timestamp, 2-D float32 array) for each calendar month.

    Opens source with ``chunks={}`` so dask uses the file's native on-disk
    chunk layout.  ``compute(scheduler='synchronous')`` then processes one
    compressed block at a time — each block is decompressed from disk
    exactly once, regardless of how many time steps it spans.

    This eliminates the catastrophic I/O pattern of the naive approach where
    a file with time-chunk=25 would decompress all 196 spatial chunks 25×
    for each time step instead of once.
    """
    time_index = pd.DatetimeIndex(da.time.values)

    for month in range(1, 13):
        idx = np.where(time_index.month == month)[0]
        if len(idx) == 0:
            continue

        month_da = da.isel(time=idx.tolist())
        if agg == 'sum':
            result = month_da.sum('time', skipna=True)
        else:
            result = month_da.mean('time', skipna=True)

        # synchronous scheduler: one dask chunk at a time → bounded RAM
        arr = result.compute(scheduler='synchronous').values
        arr = arr.astype('float32', copy=False)
        # Replace any remaining encoding artefacts (e.g. large sentinel values)
        arr[~np.isfinite(arr)] = np.float32(np.nan)

        yield pd.Timestamp(f'{year}-{month:02d}-15'), arr
        del arr


def _open_da(path, transpose_dims=None, drop_crs=False, rename_latlon=False):
    """
    Open a NetCDF file as a dask-backed DataArray with native chunk sizes.

    Parameters
    ----------
    transpose_dims  : tuple of dim names to transpose to (e.g. for OzWALD NDVI
                      which stores (time, lon, lat) instead of (time, lat, lon))
    drop_crs        : drop the 'crs' variable before converting (SILO files)
    rename_latlon   : rename lat/lon → latitude/longitude (SILO files)
    """
    ds = xr.open_dataset(path, chunks={}, mask_and_scale=True)
    if drop_crs and 'crs' in ds:
        ds = ds.drop_vars('crs')
    if rename_latlon:
        rn = {}
        if 'lat' in ds.dims and 'latitude' not in ds.dims:
            rn['lat'] = 'latitude'
        if 'lon' in ds.dims and 'longitude' not in ds.dims:
            rn['lon'] = 'longitude'
        if rn:
            ds = ds.rename(rn)
    if transpose_dims is not None:
        ds = ds.transpose(*transpose_dims)
    da = ds.to_array().squeeze('variable', drop=True)
    return da


# ---------------------------------------------------------------------------
# MODIS surface reflectance → NDWI and kNDVI
# ---------------------------------------------------------------------------

def _modis_indices(var, year, results, template, mask, verbose=False):
    """
    Compute NDWI or kNDVI from MODIS MCD43A4 surface reflectance bands.

    The spectral index is computed lazily in the dask graph so that both
    bands and the index calculation are fused into a single pass through
    the data, reading each compressed block from disk only once.
    """
    out_file = f'{results}{var}/{var}_{TARGET_GRID}_{year}.nc'
    if os.path.exists(out_file):
        if verbose:
            print(f'  {var}: already exists, skipping')
        return
    if verbose:
        print(f'  {var}')

    b = f'{BASE_PATH}/MODIS/mosaic/MCD43A4.006'
    if var == 'NDWI':
        da_a = _open_da(f'{b}/MCD43A4.006.b02.500m_0841_0876nm_nbar.{year}.nc')
        da_b = _open_da(f'{b}/MCD43A4.006.b05.500m_1230_1250nm_nbar.{year}.nc')
    else:  # kNDVI
        da_a = _open_da(f'{b}/MCD43A4.006.b01.500m_0620_0670nm_nbar.{year}.nc')
        da_b = _open_da(f'{b}/MCD43A4.006.b02.500m_0841_0876nm_nbar.{year}.nc')

    # Lazy index computation — fused with I/O inside dask graph
    with np.errstate(invalid='ignore', divide='ignore'):
        if var == 'NDWI':
            index = (da_a - da_b) / (da_a + da_b)
        else:
            ndvi = (da_b - da_a) / (da_b + da_a)
            index = xr.apply_ufunc(np.tanh, ndvi ** 2, dask='parallelized',
                                   output_dtypes=[float])

    lat_name, lon_name = _spatial_dim_names(da_a)
    src_lat = da_a.coords[lat_name].values
    src_lon = da_a.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, Resampling.average)

    out_lat, out_lon = _template_coords(template)
    mask_arr = mask.values

    os.makedirs(f'{results}{var}', exist_ok=True)
    with _nc_writer(out_file, var, out_lat, out_lon) as nc_ds:
        for t_idx, (ts, val) in enumerate(_monthly_means_gen(index, year, agg='mean')):
            # val is (lat, lon) – same layout as src_lat/src_lon
            reprojected = warp(val)
            del val
            reprojected = np.where(mask_arr, reprojected, np.float32(np.nan))
            _write_slice(nc_ds, var, t_idx, ts, reprojected)
            del reprojected

    gc.collect()


# ---------------------------------------------------------------------------
# OzWALD 8-day spectral indices → NDVI and LAI
# ---------------------------------------------------------------------------

def _ozwald_indices(var, year, results, template, mask, verbose=False):
    """
    Resample OzWALD 8-day NDVI or LAI to monthly means and reproject.

    OzWALD NDVI/LAI files store data as (time, longitude, latitude) —
    transposed to the standard order before processing.
    """
    out_file = f'{results}{var}/{var}_{TARGET_GRID}_{year}.nc'
    if os.path.exists(out_file):
        if verbose:
            print(f'  {var}: already exists, skipping')
        return
    if verbose:
        print(f'  {var}')

    da = _open_da(
        f'{BASE_PATH}/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc',
        transpose_dims=('time', 'latitude', 'longitude'),
    )

    lat_name, lon_name = _spatial_dim_names(da)
    src_lat = da.coords[lat_name].values
    src_lon = da.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, Resampling.average)

    out_lat, out_lon = _template_coords(template)
    mask_arr = mask.values

    os.makedirs(f'{results}{var}', exist_ok=True)
    with _nc_writer(out_file, var, out_lat, out_lon) as nc_ds:
        for t_idx, (ts, val) in enumerate(_monthly_means_gen(da, year, agg='mean')):
            reprojected = warp(val)
            del val
            reprojected = np.where(mask_arr, reprojected, np.float32(np.nan))
            _write_slice(nc_ds, var, t_idx, ts, reprojected)
            del reprojected

    gc.collect()


# ---------------------------------------------------------------------------
# MODIS MYD11A1 → LST
# ---------------------------------------------------------------------------

def _modis_LST(year, results, template, mask, verbose=False):
    """
    Process MODIS Aqua daytime LST (MYD11A1).

    QC filtering (years < 2022) is expressed as a lazy dask mask so it is
    applied per-chunk rather than loading the full QC array into memory.
    Temperature is converted from Kelvin to Celsius.
    """
    out_file = f'{results}LST/LST_{TARGET_GRID}_{year}.nc'
    if os.path.exists(out_file):
        if verbose:
            print('  LST: already exists, skipping')
        return
    if verbose:
        print('  LST')

    da_lst = _open_da(
        f'{BASE_PATH}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.{year}.nc'
    )

    if int(year) < 2022:
        da_qc = _open_da(
            f'{BASE_PATH}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.{year}.nc'
        )
        GOOD_QC = [0, 5, 17, 21, 64, 65, 81]
        # Build lazy QC mask — fused into the dask graph
        qc_good = (da_qc == GOOD_QC[0])
        for q in GOOD_QC[1:]:
            qc_good = qc_good | (da_qc == q)
        da_lst = da_lst.where(qc_good)

    # Kelvin → Celsius (lazy)
    da_lst = da_lst - 273.15

    lat_name, lon_name = _spatial_dim_names(da_lst)
    src_lat = da_lst.coords[lat_name].values
    src_lon = da_lst.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, Resampling.average)

    out_lat, out_lon = _template_coords(template)
    mask_arr = mask.values

    os.makedirs(f'{results}LST', exist_ok=True)
    with _nc_writer(out_file, 'LST', out_lat, out_lon) as nc_ds:
        for t_idx, (ts, val) in enumerate(_monthly_means_gen(da_lst, year, agg='mean')):
            reprojected = warp(val)
            del val
            reprojected = np.where(mask_arr, reprojected, np.float32(np.nan))
            _write_slice(nc_ds, 'LST', t_idx, ts, reprojected)
            del reprojected

    gc.collect()


# ---------------------------------------------------------------------------
# Static vegetation height → VegH
# ---------------------------------------------------------------------------

def _veg_height(year, results, template, mask, verbose=False):
    """
    Reproject the static VegH dataset and replicate across monthly time steps.
    Must run after NDWI (borrows its time axis).
    """
    out_file = f'{results}VegH/VegH_{TARGET_GRID}_{year}.nc'
    if os.path.exists(out_file):
        if verbose:
            print('  VegH: already exists, skipping')
        return
    if verbose:
        print('  VegH')

    da = xr.open_dataset(STATIC_FILES['VegH'])['VegH']
    lat_name, lon_name = _spatial_dim_names(da)
    src_lat = da.coords[lat_name].values
    src_lon = da.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, Resampling.average)

    vegh_2d = da.values.astype('float32', copy=False)
    vegh_2d = warp(vegh_2d)
    mask_arr = mask.values
    vegh_2d = np.where(mask_arr, vegh_2d, np.float32(np.nan))

    time_axis = xr.open_dataarray(f'{results}NDWI/NDWI_{TARGET_GRID}_{year}.nc').time
    timestamps = [pd.Timestamp(t.values) for t in time_axis]

    out_lat, out_lon = _template_coords(template)
    os.makedirs(f'{results}VegH', exist_ok=True)
    with _nc_writer(out_file, 'VegH', out_lat, out_lon) as nc_ds:
        for t_idx, ts in enumerate(timestamps):
            _write_slice(nc_ds, 'VegH', t_idx, ts, vegh_2d)

    gc.collect()


# ---------------------------------------------------------------------------
# OzWALD daily meteo → Tmin, Tmax, kTavg, Tavg
# ---------------------------------------------------------------------------

def _ozwald_climate(year, results, template, mask, verbose=False):
    """
    Monthly means for Tmin, Tmax, kTavg, then Tavg = Tmin + kTavg*(Tmax-Tmin).

    Pass 1: each variable is aggregated month-by-month using dask's native
    chunk scheduler so the int16 daily 500 m files (chunks=[25, 486, 600])
    are read block-efficiently.

    Pass 2: Tavg is derived one month at a time from the already-written
    files to avoid loading three full 3-D arrays simultaneously (~10 GB).
    """
    clim_inputs = {
        'Tmin':  (f'{BASE_PATH}/OzWALD/daily/meteo/Tmin/OzWALD.Tmin.{year}.nc',
                  Resampling.bilinear),
        'Tmax':  (f'{BASE_PATH}/OzWALD/daily/meteo/Tmax/OzWALD.Tmax.{year}.nc',
                  Resampling.bilinear),
        'kTavg': (f'{BASE_PATH}/OzWALD/daily/meteo/kTavg/OzWALD.kTavg.{year}.nc',
                  Resampling.nearest),
    }

    for k, (path, resamp) in clim_inputs.items():
        out_file = f'{results}{k}/{k}_{TARGET_GRID}_{year}.nc'
        if os.path.exists(out_file):
            if verbose:
                print(f'  {k}: already exists, skipping')
            continue
        if verbose:
            print(f'  {k}')

        da = _open_da(path)
        lat_name, lon_name = _spatial_dim_names(da)
        src_lat = da.coords[lat_name].values
        src_lon = da.coords[lon_name].values
        warp = _make_warper(src_lat, src_lon, template, resamp)

        out_lat, out_lon = _template_coords(template)
        os.makedirs(f'{results}{k}', exist_ok=True)
        with _nc_writer(out_file, k, out_lat, out_lon) as nc_ds:
            for t_idx, (ts, val) in enumerate(_monthly_means_gen(da, year, agg='mean')):
                reprojected = warp(val)
                del val
                _write_slice(nc_ds, k, t_idx, ts, reprojected)
                del reprojected

        gc.collect()

    # Pass 2: Tavg — one month at a time from already-written 500 m files
    out_file = f'{results}Tavg/Tavg_{TARGET_GRID}_{year}.nc'
    if os.path.exists(out_file):
        if verbose:
            print('  Tavg: already exists, skipping')
        return
    if verbose:
        print('  Tavg')

    tmin_da  = xr.open_dataarray(f'{results}Tmin/Tmin_{TARGET_GRID}_{year}.nc')
    tmax_da  = xr.open_dataarray(f'{results}Tmax/Tmax_{TARGET_GRID}_{year}.nc')
    ktavg_da = xr.open_dataarray(f'{results}kTavg/kTavg_{TARGET_GRID}_{year}.nc')
    mask_arr = mask.values
    out_lat, out_lon = _template_coords(template)

    os.makedirs(f'{results}Tavg', exist_ok=True)
    with _nc_writer(out_file, 'Tavg', out_lat, out_lon) as nc_ds:
        for t_idx in range(len(tmin_da.time)):
            ts      = pd.Timestamp(tmin_da.time.values[t_idx])
            tmin_s  = tmin_da.isel(time=t_idx).values.astype('float32', copy=False)
            tmax_s  = tmax_da.isel(time=t_idx).values.astype('float32', copy=False)
            ktavg_s = ktavg_da.isel(time=t_idx).values.astype('float32', copy=False)
            tavg_s  = tmin_s + ktavg_s * (tmax_s - tmin_s)
            tavg_s  = np.where(mask_arr, tavg_s, np.float32(np.nan))
            del tmin_s, tmax_s, ktavg_s
            _write_slice(nc_ds, 'Tavg', t_idx, ts, tavg_s)
            del tavg_s

    gc.collect()


# ---------------------------------------------------------------------------
# SILO → SRAD, rain, VPD
# ---------------------------------------------------------------------------

def _SILO_climate(year, results, template, mask, verbose=False,
                  prior_year_rain_only=False):
    """
    Monthly aggregation of SILO climate grids.

    SILO files use lat/lon dimension names and carry a 'crs' variable;
    both are handled transparently by _open_da().  Chunks are [1, 681, 841]
    so each time step is already one chunk — no redundant I/O issue here,
    but dask is still used for consistency and scale_factor handling.

    VPD = sat_vp(Tavg) - vp; Tavg is read one slice at a time.
    """
    clim_inputs = {
        'SRAD': (f'{BASE_PATH}/SILO/radiation/{year}.radiation.nc', 'mean'),
        'rain': (f'{BASE_PATH}/SILO/daily_rain/{year}.daily_rain.nc', 'sum'),
        'VPD':  (f'{BASE_PATH}/SILO/vp/{year}.vp.nc', 'mean'),
    }

    mask_arr = mask.values
    out_lat, out_lon = _template_coords(template)

    for k, (path, agg) in clim_inputs.items():
        if prior_year_rain_only and k != 'rain':
            continue

        out_file = f'{results}{k}/{k}_{TARGET_GRID}_{year}.nc'
        if os.path.exists(out_file):
            if verbose:
                print(f'  {k}: already exists, skipping')
            continue
        if verbose:
            print(f'  {k}')

        da = _open_da(path, drop_crs=True, rename_latlon=True)
        lat_name, lon_name = _spatial_dim_names(da)
        src_lat = da.coords[lat_name].values
        src_lon = da.coords[lon_name].values
        warp = _make_warper(src_lat, src_lon, template, Resampling.bilinear)

        tavg_da = None
        if k == 'VPD':
            tavg_da = xr.open_dataarray(
                f'{results}Tavg/Tavg_{TARGET_GRID}_{year}.nc'
            )

        os.makedirs(f'{results}{k}', exist_ok=True)
        with _nc_writer(out_file, k, out_lat, out_lon) as nc_ds:
            for t_idx, (ts, val) in enumerate(_monthly_means_gen(da, year, agg=agg)):
                reprojected = warp(val)
                del val

                if k == 'VPD':
                    tavg_s = tavg_da.isel(time=t_idx).values.astype('float32', copy=False)
                    sat_vp = np.float32(6.11) * np.exp(
                        np.float32(2500000 / 461) * (
                            np.float32(1 / 273) - np.float32(1.0) / (
                                np.float32(273) + tavg_s)
                        )
                    )
                    reprojected = (sat_vp - reprojected).astype('float32')
                    del tavg_s, sat_vp

                reprojected = np.where(mask_arr, reprojected, np.float32(np.nan))
                _write_slice(nc_ds, k, t_idx, ts, reprojected)
                del reprojected

        gc.collect()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def spatiotemporal_harmonisation(year, results_path=None, verbose=False):
    """
    Run all Step 1 harmonisation functions for a single year.

    Processing order respects dependencies:
      NDWI → kNDVI → NDVI → LAI → LST → VegH
      → Tmin/Tmax/kTavg → Tavg → SRAD/rain/VPD → prior-year rain
    """
    if results_path is None:
        results_path = LOCAL_RESULTS

    year = str(year)

    mask     = xr.open_dataset(STATIC_FILES['land_mask'])[f'landsea_mask_{TARGET_GRID}']
    template = mask.rio.write_crs('EPSG:4326')

    if verbose: print('Processing NDWI')
    _modis_indices('NDWI', year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing kNDVI')
    _modis_indices('kNDVI', year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing NDVI')
    _ozwald_indices('NDVI', year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing LAI')
    _ozwald_indices('LAI', year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing LST')
    _modis_LST(year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing VegH')
    _veg_height(year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing Tmin, Tmax, kTavg, Tavg')
    _ozwald_climate(year, results_path, template, mask, verbose=verbose)

    if verbose: print('Processing SRAD, rain, VPD')
    _SILO_climate(year, results_path, template, mask, verbose=verbose)

    if verbose: print(f'Processing prior-year rain ({int(year) - 1})')
    _SILO_climate(str(int(year) - 1), results_path, template, mask,
                  verbose=verbose, prior_year_rain_only=True)

    if verbose:
        print('Step 1 complete.')
        print(f'Outputs written to: {results_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 1: Spatiotemporal harmonisation.'
    )
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (e.g. 2024).')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Output directory. Default: {LOCAL_RESULTS}')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    if not check_inputs(args.year):
        sys.exit(1)

    spatiotemporal_harmonisation(
        year=args.year,
        results_path=args.results_path,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
