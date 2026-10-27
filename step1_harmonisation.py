"""
Step 1: Spatiotemporal Harmonisation of AusEFlux input datasets.

Reads raw satellite and climate data, reprojects everything to the 500 m
target grid, resamples to monthly time steps (time-stamped at the 15th of
each month), and writes one NetCDF file per variable to the interim output
directory: ``{results_path}/{var}/{var}_500m_{year}.nc``.

Each variable is skipped if its output file already exists, so the script
is safe to re-run after a partial failure.  Outputs are written atomically:
each file is streamed to ``{file}.part.{pid}`` and renamed onto its final
name only once complete (the part file is removed on any error), so an
existing final file is always complete and concurrent shards never read a
file that is still being written.  If two writers race for the same file
(e.g. ``rain_prior`` of year Y and the ``SILO`` shard of year Y-1 both
produce ``rain_{Y-1}``), the second one discards its copy once it sees the
final file.

Paths
-----
Resolved from environment variables, each overridable on the CLI:

* ``AUSEFLUX_ROOT``   (default /home/prl900/Downloads/wald_eflux)
* ``AUSEFLUX_INPUT``  (default $AUSEFLUX_ROOT/input) -> ``--input_path``
  raw inputs, mirroring the /g/data/ub8/au layout (MODIS/mosaic/..., OzWALD/..., SILO/...)
* ``AUSEFLUX_STATIC`` (default $AUSEFLUX_INPUT/static) -> ``--static_path``
  land_sea_mask_500m.nc, VegH_1km_2007_2010.nc
* output (default $AUSEFLUX_ROOT/interim_500m/) -> ``--results_path``

Memory & I/O strategy
---------------------
All source files are opened with ``xr.open_dataset(path, chunks={})`` so
that xarray uses dask with the file's **native on-disk chunk layout**.
Monthly aggregation is then expressed as a lazy dask graph and computed
with ``scheduler='synchronous'`` (default, ``--threads 1``), which processes
one compressed block at a time — reading each block from disk exactly once
instead of once per time-step (the naive approach causes > 1 TB of redundant
I/O for the daily 500 m datasets with 25-timestep time-chunks).  With
``--threads N`` (N > 1) the threaded scheduler is used with N workers (still
native chunks, so at most ~N blocks in flight) and rasterio warps with
``num_threads=N`` (both engines: odc-geo's ``rio_reproject`` forwards
``num_threads`` to ``rasterio.warp.reproject``).

Each reprojected 2-D monthly slice is written directly to the output
NetCDF file so the full 3-D time series is never held in RAM.  Output
variables are stored zlib-compressed with chunk sizes (1, 200, 200)
(time, latitude, longitude), the same layout as step2 — a pure storage-layout
choice that does not change any value.

Reprojection engines (``--engine``)
-----------------------------------
* ``rasterio`` (default): rasterio.warp.reproject directly via a
  pre-computed affine transform, avoiding repeated xarray/GDAL overhead.
* ``odc``: odc-geo ``rio_reproject`` (the kernel of ``xr_reproject``) onto a
  GeoBox built from the land-mask coordinates (same affine, same resampling
  names).  The SOURCE GeoBox is also built explicitly from ``_affine()``
  (same flips as the rasterio path) rather than inferred by odc-geo
  ``xr_reproject`` from the float32 coordinates, which gives a
  wrong pixel size on wide grids (e.g. 0.0049972534 instead of 0.005 for the
  8400-column OzWALD 500 m grids -> up to 5 px eastward shift).  Optional
  dependency, imported lazily (``uv run --with odc-geo step1_harmonisation.py
  ...``).
Both engines share all aggregation/masking/writing code; only the warp differs.

Ablation switches
-----------------
* ``--lst_qc ref`` (default): MYD11A1 QC whitelist applied for every year
  except 2022 and 2023 -- the rule of the reference ``_harmonisation.py``,
  used for all years of the published v2.1 product.  ``--lst_qc new``:
  applied only for year < 2022 (the earlier refactor rule, kept for
  comparison).
* ``--modis_nodata poison`` (default): warp NDWI/kNDVI with
  ``src_nodata=None`` so any NaN source pixel in a target block makes the
  target NaN.  This reproduces the reference's mask (its odc reprojection lost
  the nodata attribute; ~0.3 % more NaN pixel-months, Gadi experiment E4).
  ``--modis_nodata exclude``: NaN source pixels are excluded from the
  ``average`` reprojection, so a target pixel is NaN only if all its source
  pixels are NaN (arguably more correct, but it changes the wet-season
  gap-fill in the tropical north: E5 2024 north window GPP/NEE RMSE 6-10 % of
  std vs 1-3 % with ``poison``, so it is not the default).  Other variables
  are unaffected.
* ``--vars``: subset of variable groups (for PBS sharding).
* ``--orientation_check``: after writing each variable, compare the output with
  the raw source (and its vertical flip) at a few land pixels.

Usage:
    uv run step1_harmonisation.py --year 2024
    python step1_harmonisation.py --year 2024 --vars NDWI,kNDVI --threads 4
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
# Optional extra (only for --engine odc, NOT a hard dependency): "odc-geo"
#   e.g.  uv run --with odc-geo step1_harmonisation.py --year 2024 --engine odc

import argparse
import gc
import glob
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
# Constants and path resolution
# ---------------------------------------------------------------------------

TARGET_GRID = '500m'


def _env(name, default):
    v = os.environ.get(name)
    return v if v else default


ROOT_PATH = _env('AUSEFLUX_ROOT', '/home/prl900/Downloads/wald_eflux')
BASE_PATH = _env('AUSEFLUX_INPUT', f'{ROOT_PATH}/input')       # raw inputs
STATIC_PATH = _env('AUSEFLUX_STATIC', f'{BASE_PATH}/static')   # static rasters
LOCAL_RESULTS = f'{ROOT_PATH}/interim_500m/'

ALL_VARS = ['NDWI', 'kNDVI', 'NDVI', 'LAI', 'LST', 'VegH', 'climate', 'SILO',
            'rain_prior']

# Runtime options (set in main(); defaults reproduce the original behaviour)
RUN = {'engine': 'rasterio', 'threads': 1, 'lst_qc': 'ref',
       'orientation_check': False, 'modis_nodata': 'poison'}

LST_GOOD_QC = [0, 5, 17, 21, 64, 65, 81]

# Cap GDAL's internal block cache so it cannot silently consume several GB
os.environ.setdefault('GDAL_CACHEMAX', '256')   # MB


def set_paths(input_path=None, static_path=None):
    """Override the input/static roots (CLI takes precedence over env vars)."""
    global BASE_PATH, STATIC_PATH
    if input_path:
        BASE_PATH = input_path.rstrip('/')
        if not static_path and not os.environ.get('AUSEFLUX_STATIC'):
            STATIC_PATH = f'{BASE_PATH}/static'
    if static_path:
        STATIC_PATH = static_path.rstrip('/')


def static_files():
    return {
        'land_mask': f'{STATIC_PATH}/land_sea_mask_500m.nc',
        'VegH':      f'{STATIC_PATH}/VegH_1km_2007_2010.nc',
    }


def lst_qc_applies(year, rule=None):
    """Whether the MYD11A1 QC whitelist is applied for this year."""
    rule = rule or RUN['lst_qc']
    year = int(year)
    if rule == 'ref':
        return year not in (2022, 2023)   # reference _harmonisation.py (default)
    return year < 2022                    # 'new' (earlier refactor rule)


def input_files(year, lst_qc=None):
    y = str(year)
    yp = str(int(year) - 1)
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
    if lst_qc_applies(year, lst_qc):
        files['MYD11A1 QC'] = f'{b}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.{y}.nc'
    return files


def _required_labels(var, year):
    """Input-file labels (keys of input_files()) needed by a variable group."""
    yp = str(int(year) - 1)
    return {
        'NDWI':       ['MCD43A4 Band 2 (NIR)', 'MCD43A4 Band 5 (SWIR)'],
        'kNDVI':      ['MCD43A4 Band 1 (Red)', 'MCD43A4 Band 2 (NIR)'],
        'NDVI':       ['OzWALD NDVI'],
        'LAI':        ['OzWALD LAI'],
        'LST':        ['MYD11A1 LST', 'MYD11A1 QC'],   # QC only if in input_files()
        'VegH':       ['MCD43A4 Band 2 (NIR)'],        # time-axis fallback
        'climate':    ['OzWALD Tmin', 'OzWALD Tmax', 'OzWALD kTavg'],
        'SILO':       ['SILO radiation', 'SILO daily rain', 'SILO vapour pressure'],
        'rain_prior': [f'SILO daily rain {yp}'],
    }[var]


def check_inputs(year, vars_=None, lst_qc=None):
    vars_ = vars_ or ALL_VARS
    files = input_files(year, lst_qc)
    sf = static_files()
    needed = {'land_mask': sf['land_mask']}
    if 'VegH' in vars_:
        needed['VegH'] = sf['VegH']
    for v in vars_:
        for label in _required_labels(v, year):
            if label in files:
                needed[label] = files[label]
    missing = [f'{label}: {path}' for label, path in needed.items()
               if not os.path.exists(path)]
    if missing:
        print('ERROR: The following required input files are missing:')
        for m in missing:
            print(f'  {m}')
        if 'LST' in vars_ and 'MYD11A1 QC' in files and \
                not os.path.exists(files['MYD11A1 QC']):
            print(f'  (MYD11A1 QC is required for {year} under '
                  f'--lst_qc {lst_qc or RUN["lst_qc"]})')
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
    Works for both ascending and descending latitude arrays (always returns
    a north-up, west-first transform; pixel edges = centre -/+ half spacing).
    """
    dlon = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 0.005
    dlat = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 0.005
    west  = float(min(lon)) - dlon / 2
    east  = float(max(lon)) + dlon / 2
    south = float(min(lat)) - dlat / 2
    north = float(max(lat)) + dlat / 2
    return _from_bounds(west, south, east, north, len(lon), len(lat))


def _make_warper(src_lat, src_lon, template, resampling, nan_poison=False):
    """
    Pre-compute source and destination transforms once, return a
    lightweight callable::

        warp(arr_2d_float32) -> arr_2d_float32   (in template lat/lon order)

    The engine is chosen by RUN['engine'] ('rasterio' or 'odc'); both use the
    same source/destination transforms (``_affine`` of the source and
    land-mask coordinates) and resampling name.

    nan_poison=False: NaN is the source nodata (NaN pixels are excluded from
    the resampling).  nan_poison=True: no source nodata, so GDAL treats NaN as
    data and any NaN source pixel contributing to a target pixel makes it NaN
    (reference artefact, --modis_nodata poison).
    """
    lat_d, lon_d = _spatial_dim_names(template)
    dst_lat = template.coords[lat_d].values
    dst_lon = template.coords[lon_d].values
    dst_tr  = _affine(dst_lat, dst_lon)
    out_shape = (len(dst_lat), len(dst_lon))
    # The destination transform is north-up/west-first; if the template is
    # stored south-up or east-first, flip the result back to template order.
    dst_flip_lat = len(dst_lat) > 1 and dst_lat[0] < dst_lat[-1]
    dst_flip_lon = len(dst_lon) > 1 and dst_lon[0] > dst_lon[-1]

    def _to_template_order(dst):
        if dst_flip_lat:
            dst = dst[::-1, :]
        if dst_flip_lon:
            dst = dst[:, ::-1]
        return np.ascontiguousarray(dst)

    src_tr = _affine(src_lat, src_lon)
    src_nodata = None if nan_poison else np.float32(np.nan)

    # _affine() always builds a north-up, west-first transform (row 0 = north,
    # col 0 = west).  Sources whose latitude is ascending (row 0 = south, e.g.
    # SILO) or whose longitude is descending must be flipped to match, otherwise
    # they are reprojected upside-down / mirrored.  MODIS and OzWALD are
    # already north-up so are left untouched.
    src_flip_lat = len(src_lat) > 1 and src_lat[0] < src_lat[-1]
    src_flip_lon = len(src_lon) > 1 and src_lon[0] > src_lon[-1]

    def _to_source_order(arr):
        # Orient the source north-up / west-first to match src_tr and return
        # a (lat, lon) row-major float32 array.
        if src_flip_lat:
            arr = arr[::-1, :]
        if src_flip_lon:
            arr = arr[:, ::-1]
        return np.ascontiguousarray(arr.astype('float32', copy=False))

    threads = max(1, int(RUN['threads']))
    if RUN['engine'] == 'odc':
        return _make_odc_warper(src_tr, (len(src_lat), len(src_lon)),
                                out_shape, dst_tr, resampling, src_nodata,
                                _to_source_order, _to_template_order,
                                threads)

    crs = _CRS.from_epsg(4326)

    def warp(arr):
        src = _to_source_order(arr)
        dst = np.full(out_shape, np.float32(np.nan), dtype='float32')
        _rasterio_reproject(
            source=src,
            destination=dst,
            src_transform=src_tr,
            src_crs=crs,
            dst_transform=dst_tr,
            dst_crs=crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=np.float32(np.nan),
            num_threads=threads,
        )
        return _to_template_order(dst)

    return warp


def _import_odc():
    try:
        from odc.geo.geobox import GeoBox
        from odc.geo.warp import rio_reproject
    except ImportError:
        print('ERROR: --engine odc requires the optional package odc-geo, '
              'which is not installed.\n'
              '  Install it (pip install odc-geo) or run with\n'
              '  uv run --with odc-geo step1_harmonisation.py ... --engine odc')
        sys.exit(2)
    return GeoBox, rio_reproject


def _make_odc_warper(src_tr, src_shape, out_shape, dst_tr, resampling,
                     src_nodata, to_source_order, to_template_order,
                     threads=1):
    """
    odc-geo warper: the (north-up, west-first) source slice is reprojected
    with odc-geo's ``rio_reproject`` (the numpy kernel behind ``xr_reproject``)
    from an explicit source GeoBox built from ``_affine()`` (the same
    transform as the rasterio path) onto a GeoBox built from the land-mask
    coordinates (EPSG:4326, north-up).

    The source GeoBox is NOT inferred from coordinates (as ``xr_reproject``
    on a coordinate-labelled DataArray would do): odc-geo derives the pixel
    size from the first two (float32) coordinate values, which on wide grids
    is off by ~5e-7 deg per pixel (0.0049972534 instead of 0.005 for the
    8400-column OzWALD Tmin/Tmax), shifting the east of the grid by up to
    5 pixels and leaving the last columns NaN.  Passing the GeoBoxes directly
    also avoids version differences in odc-geo's coordinate handling.
    *src_nodata* None = no source nodata (NaN poisons the resampling); it is
    passed explicitly because newer odc-geo resolves a missing nodata on
    float data to NaN.  *threads* is forwarded as ``num_threads`` (odc-geo
    passes extra keyword arguments on to ``rasterio.warp.reproject``).
    """
    GeoBox, rio_reproject = _import_odc()
    src_gbox = GeoBox(src_shape, src_tr, 'EPSG:4326')
    gbox = GeoBox(out_shape, dst_tr, 'EPSG:4326')
    resampling_name = resampling.name   # 'average' | 'bilinear' | 'nearest'
    nodata = None if src_nodata is None else np.nan

    def warp(arr):
        dst = np.full(out_shape, np.float32(np.nan), dtype='float32')
        dst = rio_reproject(to_source_order(arr), dst, src_gbox, gbox,
                            resampling_name, src_nodata=nodata,
                            dst_nodata=np.nan, num_threads=threads)
        return to_template_order(np.asarray(dst, dtype='float32'))

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

PART_SUFFIX = '.part.'   # in-progress outputs: {final}.part.{pid}


def _part_files(path):
    """In-progress part files of *path* (any writer process)."""
    return glob.glob(glob.escape(path) + PART_SUFFIX + '*')


@contextmanager
def _nc_writer(path, var_name, lat, lon):
    """
    Context manager: open a netCDF4 file for streaming monthly writes.
    Yields the open Dataset; each month is written via _write_slice().
    The data variable is chunked (1, 200, 200) like step2's outputs.

    Atomic: the data are written to ``{path}.part.{pid}``, which is renamed
    onto *path* only when the block completes without error; on any error
    (including SystemExit/KeyboardInterrupt) the part file is removed.  If
    *path* appeared meanwhile (another shard wrote the same file), the part
    file is discarded and the existing file is kept.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    part = f'{path}{PART_SUFFIX}{os.getpid()}'
    ds = nc4.Dataset(part, 'w', format='NETCDF4')
    ok = False
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
                          fill_value=np.float32(np.nan), zlib=True, complevel=4,
                          chunksizes=(1, min(200, len(lat)), min(200, len(lon))))
        yield ds
        ok = True
    finally:
        ds.close()
        if not ok:
            _remove_quiet(part)
    if os.path.exists(path):
        # another writer finished the same file first: keep theirs
        print(f'  {os.path.basename(path)}: written concurrently by another '
              f'process, discarding this copy')
        _remove_quiet(part)
    else:
        os.replace(part, path)


def _remove_quiet(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _is_complete(path, var_name=None):
    """
    True if *path* is a finished output: it exists (outputs are only ever
    renamed into place when complete) and is a readable netCDF with the data
    variable and a non-empty time axis (guards against files left by older,
    non-atomic versions of this script).
    """
    if not os.path.exists(path):
        return False
    try:
        with nc4.Dataset(path) as d:
            if var_name is not None and var_name not in d.variables:
                return False
            return d.dimensions['time'].size > 0
    except Exception:
        return False


def _write_slice(nc_ds, var_name, t_idx, timestamp, data_2d):
    """Write one 2-D float32 slice to the open streaming netCDF4 dataset."""
    epoch = pd.Timestamp('1970-01-01')
    nc_ds['time'][t_idx] = float((timestamp - epoch).days)
    nc_ds[var_name][t_idx] = data_2d.astype('float32', copy=False)


# ---------------------------------------------------------------------------
# Dask-aware monthly aggregation
# ---------------------------------------------------------------------------

def _compute(obj):
    """Compute a dask-backed object with the configured scheduler."""
    n = max(1, int(RUN['threads']))
    if n > 1:
        return obj.compute(scheduler='threads', num_workers=n)
    # synchronous scheduler: one dask chunk at a time -> bounded RAM
    return obj.compute(scheduler='synchronous')


def _monthly_means_gen(da, year, agg='mean'):
    """
    Generator: yield (timestamp, 2-D float32 array) for each calendar month.

    Opens source with ``chunks={}`` so dask uses the file's native on-disk
    chunk layout.  With the synchronous scheduler each compressed block is
    decompressed from disk exactly once, regardless of how many time steps
    it spans (the threaded scheduler with --threads N keeps the same chunking,
    so at most ~N blocks are in flight).
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

        arr = _compute(result).values
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
# Orientation regression guard (--orientation_check)
# ---------------------------------------------------------------------------

def _orientation_check(out_file, var, src_da, agg='mean', n_points=20):
    """
    Cheap regression guard for the SILO flip bug (benchmark/REPORT.md,
    Finding 2).  For the first month of the written output, pick ``n_points``
    scattered land pixels, look up the nearest raw-source pixel (by lat/lon)
    and its vertical mirror (row n-1-i), aggregate the raw source over that
    month at those points only, and print corr(output, source) and
    corr(output, flipped source).  A correct output has corr≈1 vs source and
    a clearly lower value vs the flipped source.
    """
    if not RUN['orientation_check']:
        return
    try:
        out = xr.open_dataarray(out_file)
        o_lat_d, o_lon_d = _spatial_dim_names(out)
        out0 = out.isel(time=0).values
        month = pd.Timestamp(out.time.values[0]).month
        o_lat = out.coords[o_lat_d].values
        o_lon = out.coords[o_lon_d].values
        out.close()

        good = np.argwhere(np.isfinite(out0))
        if len(good) < 3:
            print(f'  [orientation] {var}: too few valid output pixels, skipped')
            return
        rng = np.random.default_rng(0)
        sel = good[rng.choice(len(good), size=min(n_points, len(good)),
                              replace=False)]
        oy, ox = sel[:, 0], sel[:, 1]
        o_vals = out0[oy, ox]

        s_lat_d, s_lon_d = _spatial_dim_names(src_da)
        s_lat = src_da.coords[s_lat_d].values.astype('float64')
        s_lon = src_da.coords[s_lon_d].values.astype('float64')
        iy = np.abs(s_lat[None, :] - o_lat[oy].astype('float64')[:, None]).argmin(1)
        ix = np.abs(s_lon[None, :] - o_lon[ox].astype('float64')[:, None]).argmin(1)
        iy_flip = len(s_lat) - 1 - iy

        def _points(rows):
            pts = src_da.isel({s_lat_d: xr.DataArray(rows, dims='p'),
                               s_lon_d: xr.DataArray(ix, dims='p')})
            if 'time' in pts.dims:
                tidx = np.where(pd.DatetimeIndex(pts.time.values).month == month)[0]
                pts = pts.isel(time=tidx.tolist())
                pts = pts.sum('time', skipna=True) if agg == 'sum' \
                    else pts.mean('time', skipna=True)
            if hasattr(pts.data, 'compute'):
                pts = _compute(pts)
            return np.asarray(pts.values, dtype='float64')

        def _corr(a, b):
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                return float('nan')
            return float(np.corrcoef(a[ok], b[ok])[0, 1])

        c = _corr(o_vals.astype('float64'), _points(iy))
        cf = _corr(o_vals.astype('float64'), _points(iy_flip))
        flag = ''
        if np.isfinite(c) and np.isfinite(cf) and cf > c:
            flag = '  <-- WARNING: output matches FLIPPED source better'
        elif np.isfinite(c) and c < 0.9:
            flag = '  <-- WARNING: low correlation with source'
        print(f'  [orientation] {var}: n={len(o_vals)} month={month:02d} '
              f'corr(out,src)={c:.4f} corr(out,flipped src)={cf:.4f}{flag}')
    except Exception as e:  # never fail the run because of the guard
        print(f'  [orientation] {var}: check failed ({e!r})')


# ---------------------------------------------------------------------------
# Shared warp + write loop
# ---------------------------------------------------------------------------

def _warp_monthly_to_file(out_file, var, src_da, year, agg, resampling,
                          template, mask_arr=None, nan_poison=False):
    """Aggregate monthly, warp each month, mask (optional) and stream-write."""
    lat_name, lon_name = _spatial_dim_names(src_da)
    src_lat = src_da.coords[lat_name].values
    src_lon = src_da.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, resampling,
                        nan_poison=nan_poison)
    out_lat, out_lon = _template_coords(template)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with _nc_writer(out_file, var, out_lat, out_lon) as nc_ds:
        for t_idx, (ts, val) in enumerate(_monthly_means_gen(src_da, year, agg=agg)):
            reprojected = warp(val)
            del val
            if mask_arr is not None:
                reprojected = np.where(mask_arr, reprojected, np.float32(np.nan))
            _write_slice(nc_ds, var, t_idx, ts, reprojected)
            del reprojected
    gc.collect()


def _skip(out_file, var, verbose):
    """Skip only complete outputs; an unreadable/empty file is regenerated."""
    if _is_complete(out_file, var):
        if verbose:
            print(f'  {var}: already exists, skipping')
        return True
    if os.path.exists(out_file):
        print(f'  WARNING: {out_file} is incomplete or unreadable '
              f'(left by an older, non-atomic run?); regenerating it')
        os.remove(out_file)
    if verbose:
        print(f'  {var}')
    return False


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
    if _skip(out_file, var, verbose):
        return

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

    # --modis_nodata poison: no source nodata, so partly-NaN source blocks
    # give NaN (reference artefact); exclude (default): NaN pixels skipped.
    _warp_monthly_to_file(out_file, var, index, year, 'mean',
                          Resampling.average, template, mask.values,
                          nan_poison=RUN['modis_nodata'] == 'poison')
    _orientation_check(out_file, var, index, 'mean')


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
    if _skip(out_file, var, verbose):
        return

    da = _open_da(
        f'{BASE_PATH}/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc',
        transpose_dims=('time', 'latitude', 'longitude'),
    )
    _warp_monthly_to_file(out_file, var, da, year, 'mean',
                          Resampling.average, template, mask.values)
    _orientation_check(out_file, var, da, 'mean')


# ---------------------------------------------------------------------------
# MODIS MYD11A1 → LST
# ---------------------------------------------------------------------------

def _modis_LST(year, results, template, mask, verbose=False):
    """
    Process MODIS Aqua daytime LST (MYD11A1).

    QC filtering (whitelist 0,5,17,21,64,65,81) is applied according to
    --lst_qc: 'new' = years < 2022 only; 'ref' = every year except 2022 and
    2023 (reference).  It is expressed as a lazy dask mask so it is applied
    per-chunk rather than loading the full QC array into memory.
    Temperature is converted from Kelvin to Celsius.
    """
    out_file = f'{results}LST/LST_{TARGET_GRID}_{year}.nc'
    if _skip(out_file, 'LST', verbose):
        return

    da_lst = _open_da(
        f'{BASE_PATH}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.{year}.nc'
    )

    if lst_qc_applies(year):
        qc_path = f'{BASE_PATH}/MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.{year}.nc'
        if not os.path.exists(qc_path):
            print(f'ERROR: MYD11A1 QC file required for {year} under '
                  f'--lst_qc {RUN["lst_qc"]} but missing: {qc_path}')
            sys.exit(1)
        if verbose:
            print(f'    applying MYD11A1 QC whitelist (--lst_qc {RUN["lst_qc"]})')
        da_qc = _open_da(qc_path)
        # Build lazy QC mask — fused into the dask graph
        qc_good = (da_qc == LST_GOOD_QC[0])
        for q in LST_GOOD_QC[1:]:
            qc_good = qc_good | (da_qc == q)
        da_lst = da_lst.where(qc_good)
    elif verbose:
        print(f'    no MYD11A1 QC for {year} (--lst_qc {RUN["lst_qc"]})')

    # Kelvin → Celsius (lazy)
    da_lst = da_lst - 273.15

    _warp_monthly_to_file(out_file, 'LST', da_lst, year, 'mean',
                          Resampling.average, template, mask.values)
    _orientation_check(out_file, 'LST', da_lst, 'mean')


# ---------------------------------------------------------------------------
# Static vegetation height → VegH
# ---------------------------------------------------------------------------

def _veg_time_axis(year, results):
    """
    Monthly time stamps for VegH: the NDWI output's time axis if a complete
    NDWI file is present (original behaviour; a file with an in-progress
    ``.part`` sibling from another shard is not used); otherwise the months
    present in the MCD43A4 band-2 source (the months NDWI would have), so
    VegH can run as its own shard.
    """
    ndwi = f'{results}NDWI/NDWI_{TARGET_GRID}_{year}.nc'
    if _is_complete(ndwi, 'NDWI') and not _part_files(ndwi):
        with xr.open_dataarray(ndwi) as t:
            return [pd.Timestamp(v) for v in t.time.values]
    b02 = (f'{BASE_PATH}/MODIS/mosaic/MCD43A4.006/'
           f'MCD43A4.006.b02.500m_0841_0876nm_nbar.{year}.nc')
    with xr.open_dataset(b02, chunks={}) as ds:
        months = sorted(set(pd.DatetimeIndex(ds.time.values).month))
    return [pd.Timestamp(f'{year}-{m:02d}-15') for m in months]


def _veg_height(year, results, template, mask, verbose=False):
    """
    Reproject the static VegH dataset and replicate across monthly time steps
    (time axis from the NDWI output, or from the MCD43A4 source if absent).
    """
    out_file = f'{results}VegH/VegH_{TARGET_GRID}_{year}.nc'
    if _skip(out_file, 'VegH', verbose):
        return

    da = xr.open_dataset(static_files()['VegH'])['VegH']
    lat_name, lon_name = _spatial_dim_names(da)
    src_lat = da.coords[lat_name].values
    src_lon = da.coords[lon_name].values
    warp = _make_warper(src_lat, src_lon, template, Resampling.average)

    vegh_2d = da.values.astype('float32', copy=False)
    vegh_2d = warp(vegh_2d)
    mask_arr = mask.values
    vegh_2d = np.where(mask_arr, vegh_2d, np.float32(np.nan))

    timestamps = _veg_time_axis(year, results)

    out_lat, out_lon = _template_coords(template)
    os.makedirs(f'{results}VegH', exist_ok=True)
    with _nc_writer(out_file, 'VegH', out_lat, out_lon) as nc_ds:
        for t_idx, ts in enumerate(timestamps):
            _write_slice(nc_ds, 'VegH', t_idx, ts, vegh_2d)

    gc.collect()
    _orientation_check(out_file, 'VegH', da, 'mean')


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
        if _skip(out_file, k, verbose):
            continue
        da = _open_da(path)
        # Tmin/Tmax/kTavg are not land-masked (as before)
        _warp_monthly_to_file(out_file, k, da, year, 'mean', resamp,
                              template, mask_arr=None)
        _orientation_check(out_file, k, da, 'mean')

    # Pass 2: Tavg — one month at a time from already-written 500 m files
    out_file = f'{results}Tavg/Tavg_{TARGET_GRID}_{year}.nc'
    if _skip(out_file, 'Tavg', verbose):
        return

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
            # Tmax/kTavg paired with Tmin by month label, not position
            tmax_s  = _month_slice(tmax_da, ts, 'Tmax')
            ktavg_s = _month_slice(ktavg_da, ts, 'kTavg')
            tavg_s  = tmin_s + ktavg_s * (tmax_s - tmin_s)
            tavg_s  = np.where(mask_arr, tavg_s, np.float32(np.nan))
            del tmin_s, tmax_s, ktavg_s
            _write_slice(nc_ds, 'Tavg', t_idx, ts, tavg_s)
            del tavg_s

    gc.collect()


def _month_slice(da, ts, name):
    """2-D float32 slice of *da* at time label *ts* (error if absent)."""
    try:
        s = da.sel(time=ts)
    except KeyError:
        s = None
    if s is None or 'time' in s.dims:
        have = ', '.join(f'{pd.Timestamp(t):%Y-%m-%d}' for t in da.time.values)
        print(f'ERROR: {name} has no time step {ts:%Y-%m-%d} (has: {have})')
        sys.exit(1)
    return s.values.astype('float32', copy=False)


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

    VPD = sat_vp(Tavg) - vp; Tavg is read one slice at a time (requires the
    Tavg output of the 'climate' group).
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
        if _skip(out_file, k, verbose):
            continue

        da = _open_da(path, drop_crs=True, rename_latlon=True)

        if k != 'VPD':
            _warp_monthly_to_file(out_file, k, da, year, agg,
                                  Resampling.bilinear, template, mask_arr)
            _orientation_check(out_file, k, da, agg)
            continue

        # VPD: needs the per-month Tavg slice
        lat_name, lon_name = _spatial_dim_names(da)
        warp = _make_warper(da.coords[lat_name].values,
                            da.coords[lon_name].values,
                            template, Resampling.bilinear)
        tavg_file = f'{results}Tavg/Tavg_{TARGET_GRID}_{year}.nc'
        if not _is_complete(tavg_file, 'Tavg') or _part_files(tavg_file):
            print(f'ERROR: VPD needs a complete {tavg_file} (missing or still '
                  f'being written by another shard); run the "climate" group '
                  f'first.')
            sys.exit(1)
        tavg_da = xr.open_dataarray(tavg_file)

        os.makedirs(f'{results}{k}', exist_ok=True)
        with _nc_writer(out_file, k, out_lat, out_lon) as nc_ds:
            for t_idx, (ts, val) in enumerate(_monthly_means_gen(da, year, agg=agg)):
                reprojected = warp(val)
                del val
                tavg_s = _month_slice(tavg_da, ts, 'Tavg')
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
        if RUN['orientation_check'] and verbose:
            print('  [orientation] VPD: skipped (derived from Tavg; '
                  'SRAD/rain cover the SILO grid orientation)')


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def spatiotemporal_harmonisation(year, results_path=None, verbose=False,
                                 vars_=None):
    """
    Run the Step 1 harmonisation functions for a single year.

    Processing order respects dependencies:
      NDWI → kNDVI → NDVI → LAI → LST → VegH
      → Tmin/Tmax/kTavg → Tavg → SRAD/rain/VPD → prior-year rain
    ``vars_`` restricts to a subset of ALL_VARS (default: all).
    """
    if results_path is None:
        results_path = LOCAL_RESULTS
    results_path = os.path.join(results_path, '')   # ensure trailing slash
    vars_ = vars_ or ALL_VARS

    year = str(year)

    if 'SILO' in vars_ and 'climate' not in vars_:
        tavg = f'{results_path}Tavg/Tavg_{TARGET_GRID}_{year}.nc'
        vpd = f'{results_path}VPD/VPD_{TARGET_GRID}_{year}.nc'
        if not _is_complete(tavg, 'Tavg') and not _is_complete(vpd, 'VPD'):
            print(f'ERROR: SILO VPD needs {tavg}; run the "climate" group first '
                  f'(or include it in --vars).')
            sys.exit(1)

    mask     = xr.open_dataset(static_files()['land_mask'])[f'landsea_mask_{TARGET_GRID}']
    template = mask.rio.write_crs('EPSG:4326')

    if verbose:
        print(f'Step 1 {year}: engine={RUN["engine"]} threads={RUN["threads"]} '
              f'lst_qc={RUN["lst_qc"]} modis_nodata={RUN["modis_nodata"]} '
              f'vars={",".join(vars_)}')
        print(f'  input={BASE_PATH} static={STATIC_PATH} results={results_path}')

    if 'NDWI' in vars_:
        if verbose: print('Processing NDWI')
        _modis_indices('NDWI', year, results_path, template, mask, verbose=verbose)

    if 'kNDVI' in vars_:
        if verbose: print('Processing kNDVI')
        _modis_indices('kNDVI', year, results_path, template, mask, verbose=verbose)

    if 'NDVI' in vars_:
        if verbose: print('Processing NDVI')
        _ozwald_indices('NDVI', year, results_path, template, mask, verbose=verbose)

    if 'LAI' in vars_:
        if verbose: print('Processing LAI')
        _ozwald_indices('LAI', year, results_path, template, mask, verbose=verbose)

    if 'LST' in vars_:
        if verbose: print('Processing LST')
        _modis_LST(year, results_path, template, mask, verbose=verbose)

    if 'VegH' in vars_:
        if verbose: print('Processing VegH')
        _veg_height(year, results_path, template, mask, verbose=verbose)

    if 'climate' in vars_:
        if verbose: print('Processing Tmin, Tmax, kTavg, Tavg')
        _ozwald_climate(year, results_path, template, mask, verbose=verbose)

    if 'SILO' in vars_:
        if verbose: print('Processing SRAD, rain, VPD')
        _SILO_climate(year, results_path, template, mask, verbose=verbose)

    if 'rain_prior' in vars_:
        if verbose: print(f'Processing prior-year rain ({int(year) - 1})')
        _SILO_climate(str(int(year) - 1), results_path, template, mask,
                      verbose=verbose, prior_year_rain_only=True)

    if verbose:
        print('Step 1 complete.')
        print(f'Outputs written to: {results_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_vars(s):
    if not s:
        return list(ALL_VARS)
    req = [v.strip() for v in s.split(',') if v.strip()]
    bad = [v for v in req if v not in ALL_VARS]
    if bad:
        raise argparse.ArgumentTypeError(
            f'unknown --vars {bad}; choose from {",".join(ALL_VARS)}')
    return [v for v in ALL_VARS if v in req]   # canonical dependency order


def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 1: Spatiotemporal harmonisation.'
    )
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (e.g. 2024).')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Raw input root (MODIS/, OzWALD/, SILO/ layout). '
                             f'Default: $AUSEFLUX_INPUT or {BASE_PATH}')
    parser.add_argument('--static_path', type=str, default=None,
                        help='Directory with land_sea_mask_500m.nc and '
                             'VegH_1km_2007_2010.nc. Default: $AUSEFLUX_STATIC '
                             f'or <input_path>/static ({STATIC_PATH})')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Output directory; files go to '
                             '{results_path}/{var}/{var}_500m_{year}.nc. '
                             f'Default: $AUSEFLUX_ROOT/interim_500m/ ({LOCAL_RESULTS})')
    parser.add_argument('--vars', type=_parse_vars, default=None,
                        help='Comma-separated subset of variable groups to '
                             f'process: {",".join(ALL_VARS)} (default: all). '
                             'climate = Tmin,Tmax,kTavg,Tavg; SILO = SRAD,rain,VPD '
                             '(VPD needs Tavg from "climate"); rain_prior = '
                             'SILO rain of year-1. VegH uses the time axis of a '
                             'complete NDWI output if present, otherwise the '
                             'MCD43A4 months.')
    parser.add_argument('--lst_qc', choices=['new', 'ref'], default='ref',
                        help='MYD11A1 QC whitelist rule: ref (default) = '
                             'every year except 2022 and 2023, the rule of the '
                             'reference _harmonisation.py used for all years '
                             'of the published v2.1 product; new = only for '
                             'year < 2022 (earlier refactor rule, kept for '
                             'comparison). A missing required QC file is an '
                             'error.')
    parser.add_argument('--engine', choices=['rasterio', 'odc'], default='rasterio',
                        help='Reprojection engine: rasterio (default) or odc '
                             '(odc-geo rio_reproject between explicit source/'
                             'land-mask GeoBoxes; needs optional odc-geo).')
    parser.add_argument('--modis_nodata', choices=['exclude', 'poison'],
                        default='poison',
                        help='NaN handling in the average reprojection of the '
                             'MODIS indices NDWI/kNDVI: poison (default) = no '
                             'source nodata, so any NaN source pixel in a '
                             'target block makes it NaN, reproducing the '
                             'reference mask (~0.3 %% more NaN pixel-months); '
                             'exclude = NaN source pixels are skipped (moves '
                             'wet-season north NEE by 1-2 %%/month vs v2.1).')
    parser.add_argument('--threads', type=int, default=1,
                        help='Threads (>= 1) for dask (threaded scheduler '
                             'when > 1, else synchronous) and the warp '
                             'num_threads (rasterio and odc engines). '
                             'Default 1.')
    parser.add_argument('--orientation_check', action='store_true', default=False,
                        help='After writing each variable, print corr(output, '
                             'raw source) and corr(output, vertically flipped '
                             'source) at a few land pixels (regression guard '
                             'for the SILO flip bug).')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be >= 1')

    set_paths(args.input_path, args.static_path)
    RUN['engine'] = args.engine
    RUN['threads'] = args.threads
    RUN['lst_qc'] = args.lst_qc
    RUN['orientation_check'] = args.orientation_check
    RUN['modis_nodata'] = args.modis_nodata
    vars_ = args.vars or list(ALL_VARS)

    if args.engine == 'odc':
        _import_odc()   # fail early with a clear message

    if not check_inputs(args.year, vars_, args.lst_qc):
        sys.exit(1)

    spatiotemporal_harmonisation(
        year=args.year,
        results_path=args.results_path,
        verbose=args.verbose,
        vars_=vars_,
    )


if __name__ == '__main__':
    main()
