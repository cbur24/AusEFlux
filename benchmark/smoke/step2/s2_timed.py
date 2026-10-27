"""
Step 2: Feature Dataset Construction for AusEFlux.

Combines per-year interim files from Step 1 into multi-year feature datasets,
applies gap-filling, computes vegetation fractions (Donohue et al. 2009),
cumulative rainfall, fractional anomalies, ΔT, and C4 grass fraction.

Memory strategy: data is processed in spatial tiles (latitude bands) with the
full time series loaded for each band.  Peak RSS stays under ~4 GB even for
20+ years of data on the 500 m grid (per in-flight tile; see --threads).

Stages (run in this order; --stages selects a subset):
  gapfill  interim/{var}/*.nc -> {var}_500m.nc (rain: rain_500m.nc trimmed to
           the first non-rain year, plus rain_500m_extrayear.nc for cmlrain)
  vegfrac  gap-filled NDVI + static ndvi_of_baresoil -> trees/grass/bare
  cmlrain  gap-filled rain (incl. the extra prior year) -> rain_cml{3,6,12}
  deltat   gap-filled LST and Tavg -> ΔT
  anom     gap-filled vars + rain + rain_cml* -> {var}_anom
  c4       grass + static C4 tif + land mask -> C4_grass

Any history a stage needs that is not available as a full-length output file
(e.g. because an earlier stage ran with --years_out) is recomputed inline from
the interim files, so outputs are bit-identical to a full run.

Monthly climatologies (2003-2022) are persisted under
{results_path}/climatology/ and reused on later runs (--reuse_clim):
  {var}_gapfill_clim_500m.nc   climatology of the raw interim var (gap-filling)
  {var}_clim_500m.nc           climatology of the anomaly input (anom stage)
They must be deleted (or --no_reuse_clim used) if the 2003-2022 inputs change.

Paths: AUSEFLUX_ROOT (default /home/prl900/Downloads/wald_eflux),
AUSEFLUX_INPUT (default $AUSEFLUX_ROOT/input), AUSEFLUX_STATIC (default
$AUSEFLUX_INPUT/static); interim default $AUSEFLUX_ROOT/interim_500m, results
default $AUSEFLUX_ROOT/features_500m.  All overridable on the command line.

Usage:
    uv run step2_feature_datasets.py
    uv run step2_feature_datasets.py --interim_path interim_500m/ --results_path features_500m/
    uv run step2_feature_datasets.py --stages gapfill --vars NDWI      # one PBS array task
    uv run step2_feature_datasets.py --years_out 2024-2025 --threads 8  # annual update
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
#     "bottleneck",
#     "flox",
# ]
# ///

import argparse
import gc
import multiprocessing as mp
import os
import sys
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor

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


def _env(name, default):
    """Environment variable *name*, or *default* if unset/empty."""
    v = os.environ.get(name)
    return v if v else default


AUSEFLUX_ROOT   = _env('AUSEFLUX_ROOT', '/home/prl900/Downloads/wald_eflux')
AUSEFLUX_INPUT  = _env('AUSEFLUX_INPUT', os.path.join(AUSEFLUX_ROOT, 'input'))
AUSEFLUX_STATIC = _env('AUSEFLUX_STATIC', os.path.join(AUSEFLUX_INPUT, 'static'))

BASE_PATH = AUSEFLUX_ROOT
DEFAULT_INTERIM = os.path.join(AUSEFLUX_ROOT, 'interim_500m')
DEFAULT_RESULTS = os.path.join(AUSEFLUX_ROOT, 'features_500m')
DEFAULT_STATIC  = AUSEFLUX_STATIC
TARGET_GRID = '500m'

EXCLUDE_VARS = {'Tmin', 'Tmax', 'kTavg'}
INTERP_GAPFILL_VARS = {'NDWI', 'kNDVI'}

CLIM_START = '2003'
CLIM_END   = '2022'

ANOM_VARS = ['NDWI', 'SRAD', 'Tavg', 'VPD', 'kNDVI', 'LAI',
             'rain', 'rain_cml3', 'rain_cml6', 'rain_cml12']
CML_WINDOWS = [(3, 'rain_cml3'), (6, 'rain_cml6'), (12, 'rain_cml12')]

STAGES = ['gapfill', 'vegfrac', 'cmlrain', 'deltat', 'anom', 'c4']

MAX_TILE_MEM = 2.0e9  # ~2 GB per tile array
CHUNK_ROWS = 200     # output chunk size (1, 200, 200)

# Global attribute marking an output written with --years_out (truncated).
YEARS_OUT_ATTR = 'auseflux_years_out'

os.environ.setdefault('GDAL_CACHEMAX', '256')

# netCDF4/HDF5 is not thread-safe: every read and write in this script goes
# through this lock; only the numerical work runs concurrently.
_IO_LOCK = threading.RLock()


class Config:
    """Run configuration shared by all stages."""

    def __init__(self, interim_path, results_path, static_path,
                 years_out=None, bare_clip=True, anom_guard='one',
                 vars_=None, reuse_clim=True, threads=1, pool='process',
                 verbose=False):
        self.interim_path = interim_path
        self.results_path = results_path
        self.static_path = static_path
        self.years_out = years_out          # (y0, y1) or None
        self.bare_clip = bare_clip
        self.anom_guard = anom_guard
        self.vars = vars_                   # set or None (= all)
        self.reuse_clim = reuse_clim
        self.threads = max(1, int(threads))
        self.pool = pool
        self.verbose = verbose

    @property
    def years_out_str(self):
        if self.years_out is None:
            return None
        return f'{self.years_out[0]}-{self.years_out[1]}'

    def out_mask(self, times, start=None):
        """Boolean mask of *times* to write (>= *start* year, within --years_out)."""
        idx = pd.DatetimeIndex(times)
        m = np.ones(len(idx), dtype=bool)
        if start is not None:
            m &= idx >= pd.Timestamp(f'{start}-01-01')
        if self.years_out is not None:
            m &= (idx.year >= self.years_out[0]) & (idx.year <= self.years_out[1])
        return m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TILE_ROWS_OVERRIDE = None   # --tile_rows (testing / tuning)


def _tile_rows(n_times, n_lon, max_mem=MAX_TILE_MEM):
    """Number of latitude rows per tile to stay under *max_mem* bytes."""
    if _TILE_ROWS_OVERRIDE:
        return _TILE_ROWS_OVERRIDE
    bytes_per_row = n_times * n_lon * 4
    rows = int(max_mem / bytes_per_row) if bytes_per_row > 0 else 400
    rows = min(max(rows, 50), 400)
    # Align tiles with the 200-row output chunks: a tile that splits a chunk
    # forces HDF5 to re-read, decompress and re-compress it (results are
    # independent of the tile size).
    if rows >= CHUNK_ROWS:
        return rows // CHUNK_ROWS * CHUNK_ROWS
    return 100 if rows >= 100 else 50


_CLIM_REPORTED = set()


def _clim_subset(da, label=''):
    """
    Select the climatology reference period (2003–2022) from *da*.
    Falls back to all available data if the reference period has no overlap
    (deterministic: every tile of a variable has the same time axis).
    """
    subset = da.sel(time=slice(CLIM_START, CLIM_END))
    if subset.sizes['time'] == 0:
        if label not in _CLIM_REPORTED:
            _CLIM_REPORTED.add(label)
            yrs = pd.DatetimeIndex(da.time.values).year
            print(f'    NOTE {label}: no data in {CLIM_START}-{CLIM_END}; '
                  f'climatology falls back to all data {yrs.min()}-{yrs.max()}')
        return da
    if label not in _CLIM_REPORTED:
        _CLIM_REPORTED.add(label)
        yrs = pd.DatetimeIndex(subset.time.values).year
        print(f'    {label}: climatology from {yrs.min()}-{yrs.max()}')
    return subset


def _clim_years(times):
    """Years actually used for the climatology of a series with *times*."""
    idx = pd.DatetimeIndex(times)
    m = (idx >= pd.Timestamp(f'{CLIM_START}-01-01')) & \
        (idx < pd.Timestamp(f'{int(CLIM_END) + 1}-01-01'))
    yrs = idx[m].year if m.any() else idx.year
    return f'{yrs.min()}-{yrs.max()}'


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


def _isel_index(idx):
    """Index array -> slice if contiguous (cheaper for dask), else the array."""
    if idx is None:
        return slice(None)
    idx = np.asarray(idx)
    if idx.size and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _out_path(cfg, name):
    return os.path.join(cfg.results_path, f'{name}_{TARGET_GRID}.nc')


def _file_years_out(path):
    """The --years_out a file was written with (None = full length)."""
    with _IO_LOCK, nc4.Dataset(path) as d:
        return getattr(d, YEARS_OUT_ATTR, None)


def _usable(cfg, path, need_full):
    """
    Can an existing output *path* serve as input to a later stage?
    need_full: the stage needs the full history (rolling windows,
    climatologies); truncated files are then never usable.  Otherwise a
    truncated file is usable if it was written with the same --years_out.
    """
    if not os.path.exists(path):
        return False
    yo = _file_years_out(path)
    if yo is None:
        return True
    return (not need_full) and yo == cfg.years_out_str


def _exists(cfg, path, label):
    """Skip-if-exists check (warns if written with a different --years_out)."""
    if not os.path.exists(path):
        return False
    yo = _file_years_out(path)
    if yo != cfg.years_out_str and cfg.verbose:
        print(f'  WARNING {label}: existing file has years_out={yo}, '
              f'requested {cfg.years_out_str}; keeping it (delete to redo)')
    if cfg.verbose:
        print(f'  {label}: exists, skipping')
    return True


_WT = [0.0]
_JOB = None   # tile compute function inherited by forked workers


def _job_call(lat_s, lat_e):
    return _JOB(lat_s, lat_e)


def _child_init():
    """Forked worker: drop inherited file handles; xarray reopens lazily."""
    try:
        from xarray.backends.file_manager import FILE_CACHE
        FILE_CACHE.clear()
    except Exception:
        pass


def _run_tiles(cfg, n_lat, tile_r, compute, write, label=''):
    """
    Run compute(lat_s, lat_e) -> result over latitude tiles and pass each
    result to write(lat_s, lat_e, result) in tile order, from the calling
    process/thread (netCDF4 writes are not thread/process safe).

    cfg.threads > 1 computes up to cfg.threads tiles concurrently, at most
    cfg.threads tiles in flight (bounded memory).  cfg.pool 'process' uses
    forked worker processes (the compute closure is inherited, no pickling of
    sources; output files are opened lazily, after the fork); 'thread' uses a
    thread pool (little gain: xarray groupby/interpolate_na hold the GIL).
    """
    global _JOB
    starts = [(s, min(s + tile_r, n_lat)) for s in range(0, n_lat, tile_r)]
    every = max(1, len(starts) // 10)

    def _done(i, s, e, res):
        _t = time.time(); write(s, e, res); _WT[0] += time.time() - _t
        del res
        gc.collect()
        if cfg.verbose and len(starts) > 1 and (i % every == 0):
            print(f'    {label} rows {s}–{e} / {n_lat}')

    n = min(cfg.threads, len(starts))
    if n <= 1:
        for i, (s, e) in enumerate(starts):
            _done(i, s, e, compute(s, e))
        return

    if cfg.pool == 'process':
        _JOB = compute
        try:
            with mp.get_context('fork').Pool(n, initializer=_child_init) as pool:
                pending = deque()
                for i, (s, e) in enumerate(starts):
                    pending.append((i, s, e,
                                    pool.apply_async(_job_call, (s, e))))
                    if len(pending) >= n:
                        j, ps, pe, r = pending.popleft()
                        _done(j, ps, pe, r.get())
                while pending:
                    j, ps, pe, r = pending.popleft()
                    _done(j, ps, pe, r.get())
        finally:
            _JOB = None
        return

    with ThreadPoolExecutor(max_workers=n) as ex:
        pending = deque()
        for i, (s, e) in enumerate(starts):
            pending.append((i, s, e, ex.submit(compute, s, e)))
            if len(pending) >= n:
                j, ps, pe, fut = pending.popleft()
                _done(j, ps, pe, fut.result())
        while pending:
            j, ps, pe, fut = pending.popleft()
            _done(j, ps, pe, fut.result())


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

class _OutFile:
    """
    Pre-allocated (time, lat, lon) float32 NetCDF written tile by tile.

    Written to ``path + '.part'`` and renamed on finish(), so an interrupted
    run never leaves a complete-looking file behind (skip-if-exists stays
    safe).  Files restricted by --years_out carry a global attribute.
    """

    def __init__(self, path, var_name, lat, lon, times, years_out=None):
        # Created lazily on first write: worker processes are forked before
        # any output file is opened.
        self.path = path
        self.tmp = path + '.part'
        self.var_name = var_name
        self._spec = (lat, lon, times, years_out)
        self.ds = None

    def _open(self):
        if self.ds is not None:
            return
        lat, lon, times, years_out = self._spec
        var_name = self.var_name
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with _IO_LOCK:
            ds = nc4.Dataset(self.tmp, 'w', format='NETCDF4')
            self.ds = ds
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
                chunksizes=(1, min(CHUNK_ROWS, len(lat)), min(200, len(lon))),
            )
            if years_out is not None:
                setattr(ds, YEARS_OUT_ATTR, years_out)

    def write(self, lat_s, lat_e, arr, t=None):
        self._open()
        with _IO_LOCK:
            if t is None:
                self.ds[self.var_name][:, lat_s:lat_e, :] = arr
            else:
                self.ds[self.var_name][t, lat_s:lat_e, :] = arr

    def finish(self):
        self._open()
        with _IO_LOCK:
            self.ds.close()
        os.replace(self.tmp, self.path)

    def abort(self):
        try:
            if self.ds is not None:
                with _IO_LOCK:
                    self.ds.close()
        finally:
            if os.path.exists(self.tmp):
                os.remove(self.tmp)


class _Outputs:
    """Context manager owning several _OutFile objects (finish/abort all)."""

    def __init__(self):
        self.files = {}

    def add(self, key, *args, **kw):
        self.files[key] = _OutFile(*args, **kw)
        return self.files[key]

    def __getitem__(self, key):
        return self.files[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            for f in self.files.values():
                f.finish()
        else:
            for f in self.files.values():
                f.abort()
        return False


# ---------------------------------------------------------------------------
# Persisted monthly climatologies
# ---------------------------------------------------------------------------

class _ClimStore:
    """
    Monthly climatology (month, lat, lon) persisted at
    {results}/climatology/{name}_clim_500m.nc.  Values are stored with the
    dtype they were computed in, so reuse is bit-identical to recomputing.
    """

    def __init__(self, cfg, name, lat, lon):
        self.cfg = cfg
        self.name = name
        self.lat, self.lon = lat, lon
        self.path = os.path.join(cfg.results_path, 'climatology',
                                 f'{name}_clim_{TARGET_GRID}.nc')
        self.ds = None
        self.writer = None
        self.available = False
        if cfg.reuse_clim and os.path.exists(self.path):
            with _IO_LOCK:
                ds = xr.open_dataset(self.path)
                ok = (int(ds.attrs.get('use_flox', 0))
                      == int(bool(xr.get_options()['use_flox']))
                      and ds.sizes.get('latitude') == len(lat)
                      and ds.sizes.get('longitude') == len(lon)
                      and np.array_equal(ds.latitude.values.astype('float32'), lat)
                      and np.array_equal(ds.longitude.values.astype('float32'), lon))
            if ok:
                self.ds = ds
                self.months = ds.month.values
                self.available = True
                if cfg.verbose:
                    print(f'    reusing climatology {self.path} '
                          f'(years {ds.attrs.get("clim_years", "?")})')
            else:
                ds.close()
                print(f'    WARNING: {self.path} does not match the grid '
                      f'(or --flox setting); recomputing')

    def read(self, lat_s, lat_e, tile):
        """Climatology rows lat_s:lat_e with *tile*'s spatial coordinates."""
        with _IO_LOCK:
            arr = self.ds['clim'].isel(latitude=slice(lat_s, lat_e)).values
        return xr.DataArray(arr, dims=('month', 'latitude', 'longitude'),
                            coords={'month': self.months,
                                    'latitude': tile.latitude.values,
                                    'longitude': tile.longitude.values})

    def write(self, lat_s, lat_e, clim, clim_years):
        """Write freshly computed climatology rows (main thread only)."""
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + '.part'
            with _IO_LOCK:
                d = nc4.Dataset(tmp, 'w', format='NETCDF4')
                months = clim['month'].values
                d.createDimension('month', len(months))
                d.createDimension('latitude', len(self.lat))
                d.createDimension('longitude', len(self.lon))
                mv = d.createVariable('month', 'i4', ('month',))
                mv[:] = months
                lv = d.createVariable('latitude', 'f4', ('latitude',))
                lv.units = 'degrees_north'
                lv[:] = self.lat
                xv = d.createVariable('longitude', 'f4', ('longitude',))
                xv.units = 'degrees_east'
                xv[:] = self.lon
                d.createVariable('clim', clim.dtype,
                                 ('month', 'latitude', 'longitude'),
                                 zlib=True, complevel=4,
                                 chunksizes=(1, min(200, len(self.lat)),
                                             min(200, len(self.lon))))
                d.clim_start = CLIM_START
                d.clim_end = CLIM_END
                d.clim_years = clim_years
                d.source = self.name
                d.use_flox = int(bool(xr.get_options()['use_flox']))
            self.writer = (d, tmp)
        d, _ = self.writer
        with _IO_LOCK:
            d['clim'][:, lat_s:lat_e, :] = clim.transpose(
                'month', 'latitude', 'longitude').values

    def finish(self, ok=True):
        if self.ds is not None:
            with _IO_LOCK:
                self.ds.close()
            self.ds = None
        if self.writer is not None:
            d, tmp = self.writer
            with _IO_LOCK:
                d.close()
            if ok:
                os.replace(tmp, self.path)
                if self.cfg.verbose:
                    print(f'    saved climatology {self.path}')
            elif os.path.exists(tmp):
                os.remove(tmp)
            self.writer = None


# ---------------------------------------------------------------------------
# Tileable sources: full-history float32 (time, lat, lon) series
# ---------------------------------------------------------------------------
#
# Every source exposes .name, .times (DatetimeIndex), .lat/.lon (output
# coordinates), .n_lat/.n_lon and .read(lat_s, lat_e, t_idx=None) returning a
# float32 DataArray holding exactly the values the corresponding output file
# holds (or would hold in a full run).

class _FileSource:
    """An existing step-2 output file."""

    def __init__(self, path):
        self.path = path
        with _IO_LOCK:
            self.ds = xr.open_dataset(path, chunks={})
            self.name = _data_var(self.ds)
            self.da = self.ds[self.name]
        self.times = pd.DatetimeIndex(self.da.time.values)
        self.lat, self.lon = _coords(self.da)
        self.n_lat, self.n_lon = len(self.lat), len(self.lon)

    def read(self, lat_s, lat_e, t_idx=None):
        with _IO_LOCK:
            tile = self.da.isel(latitude=slice(lat_s, lat_e),
                                time=_isel_index(t_idx)).compute(
                scheduler='synchronous')
        return tile.astype('float32', copy=False)

    def close(self):
        with _IO_LOCK:
            self.ds.close()


class _GapfillSource:
    """Gap-filled variable computed on the fly from the interim files."""

    def __init__(self, cfg, var, clim_store=None):
        self.cfg = cfg
        self.var = var
        files = _nc_files(os.path.join(cfg.interim_path, var))
        if not files:
            raise FileNotFoundError(f'no interim files for {var}')
        with _IO_LOCK:
            self.ds = xr.open_mfdataset(files, chunks={})
            self.name = _data_var(self.ds)
            self.da = self.ds[self.name]
        self.times = pd.DatetimeIndex(self.da.time.values)
        self.lat, self.lon = _coords(self.da)
        self.n_lat, self.n_lon = len(self.lat), len(self.lon)
        self.clim = clim_store if (clim_store is not None
                                   and clim_store.available) else None

    def compute(self, lat_s, lat_e, t_idx=None):
        """
        Return (filled, fresh_clim): the gap-filled float32 tile restricted
        to *t_idx*, and the climatology if it was computed (None if reused).

        With a reused climatology the non-interpolated variables only read
        the requested time steps (fillna is element-wise); NDWI/kNDVI always
        read the full history (linear interpolation across gaps).
        """
        windowed = (self.clim is not None and t_idx is not None
                    and self.var not in INTERP_GAPFILL_VARS)
        with _IO_LOCK:
            tile = self.da.isel(
                latitude=slice(lat_s, lat_e),
                time=_isel_index(t_idx) if windowed else slice(None),
            ).compute(scheduler='synchronous')

        if self.clim is not None:
            clim = self.clim.read(lat_s, lat_e, tile)
            fresh = None
        else:
            clim = _clim_subset(tile, f'{self.var} (gapfill)') \
                       .groupby('time.month').mean()
            fresh = clim

        if self.var in INTERP_GAPFILL_VARS:
            anom = tile.groupby('time.month') - clim
            anom = anom.interpolate_na(
                dim='time', method='linear', limit=2)
            filled = anom.groupby('time.month') + clim
            filled = filled.drop_vars('month')
            filled = filled.groupby('time.month').fillna(clim)
            filled = filled.drop_vars('month')
        else:
            filled = tile.groupby('time.month').fillna(clim)
            filled = filled.drop_vars('month')

        filled = filled.astype('float32', copy=False)
        if not windowed and t_idx is not None:
            filled = filled.isel(time=_isel_index(t_idx))
        return filled, fresh

    def read(self, lat_s, lat_e, t_idx=None):
        return self.compute(lat_s, lat_e, t_idx)[0]

    def close(self):
        with _IO_LOCK:
            self.ds.close()
        if self.clim is not None:
            self.clim.finish()


class _TimeView:
    """A time subset (boolean *mask*) of another source."""

    def __init__(self, src, mask):
        self.src = src
        self.parent_idx = np.where(mask)[0]
        self.name = src.name
        self.times = src.times[mask]
        self.lat, self.lon = src.lat, src.lon
        self.n_lat, self.n_lon = src.n_lat, src.n_lon

    def read(self, lat_s, lat_e, t_idx=None):
        idx = self.parent_idx if t_idx is None else self.parent_idx[t_idx]
        return self.src.read(lat_s, lat_e, idx)

    def close(self):
        self.src.close()


class _CmlSource:
    """rain_cml{window} computed on the fly from the full rain history."""

    def __init__(self, rain_hist, window, name):
        self.src = rain_hist
        self.window = window
        self.name = name
        mask = rain_hist.times >= pd.Timestamp(f'{CLIM_START}-01-01')
        self.base_idx = np.where(mask)[0]
        self.times = rain_hist.times[mask]
        self.lat, self.lon = rain_hist.lat, rain_hist.lon
        self.n_lat, self.n_lon = rain_hist.n_lat, rain_hist.n_lon

    def read(self, lat_s, lat_e, t_idx=None):
        tile = self.src.read(lat_s, lat_e)
        cml = tile.rolling(time=self.window, min_periods=self.window).sum()
        idx = self.base_idx if t_idx is None else self.base_idx[t_idx]
        return cml.isel(time=idx).astype('float32', copy=False)

    def close(self):
        self.src.close()


def _first_year(cfg):
    """First year of the non-rain variables (rain is trimmed to it)."""
    vars_all = _feature_vars(cfg.interim_path)
    ref_var = next((v for v in vars_all if v != 'rain'), vars_all[0])
    ref_files = _nc_files(os.path.join(cfg.interim_path, ref_var))
    with _IO_LOCK, xr.open_mfdataset(ref_files) as ref_ds:
        return int(pd.DatetimeIndex(ref_ds.time.values).year.min())


def _gapfill_clim_store(cfg, var, src_like):
    return _ClimStore(cfg, f'{var}_gapfill', src_like.lat, src_like.lon)


def _new_gapfill_source(cfg, var):
    """Inline gap-fill source, reusing a stored gap-fill climatology."""
    src = _GapfillSource(cfg, var)
    store = _gapfill_clim_store(cfg, var, src)
    if store.available:
        src.clim = store
    return src


def _feature_source(cfg, var, need_full=True):
    """Gap-filled *var*: output file if usable, else computed inline."""
    path = _out_path(cfg, var)
    if _usable(cfg, path, need_full):
        return _FileSource(path)
    if cfg.verbose:
        print(f'    {var}: computing gap-filled history inline from interim')
    return _new_gapfill_source(cfg, var)


def _rain_hist_source(cfg):
    """Full gap-filled rain history (including the extra prior year)."""
    path = os.path.join(cfg.results_path, f'rain_{TARGET_GRID}_extrayear.nc')
    if _usable(cfg, path, need_full=True):
        return _FileSource(path)
    if cfg.verbose:
        print('    rain: computing gap-filled history inline from interim')
    return _new_gapfill_source(cfg, 'rain')


def _anom_input_source(cfg, var, need_full):
    """The series whose fractional anomaly is computed (as in a full run)."""
    path = _out_path(cfg, var)
    if _usable(cfg, path, need_full):
        return _FileSource(path)
    if cfg.verbose:
        print(f'    {var}: computing history inline')
    if var == 'rain':
        src = _new_gapfill_source(cfg, 'rain')
        return _TimeView(src, src.times.year >= _first_year(cfg))
    for window, name in CML_WINDOWS:
        if var == name:
            return _CmlSource(_rain_hist_source(cfg), window, name)
    return _new_gapfill_source(cfg, var)


# ---------------------------------------------------------------------------
# Step 2a: Combine per-year files + gap-fill
# ---------------------------------------------------------------------------

def _combine_gapfill(cfg):
    """
    For each variable, combine annual interim files into a single multi-year
    feature file with gap-filling.  Rain is written as ``rain_500m.nc``
    (trimmed to the first non-rain year) and, in full runs while any
    cumulative-rain output is missing, as ``rain_500m_extrayear.nc``.
    """
    vars_all = _feature_vars(cfg.interim_path)
    vars_to_process = [v for v in vars_all
                       if cfg.vars is None or v in cfg.vars]
    if not vars_to_process:
        print('  no gap-fill variables selected')
        return

    for var in vars_to_process:
        # --- which outputs are missing -------------------------------------
        wanted = []   # (key, path, start_year_or_None)
        if var == 'rain':
            rain_path = _out_path(cfg, 'rain')
            ext_path = os.path.join(cfg.results_path,
                                    f'rain_{TARGET_GRID}_extrayear.nc')
            cml_missing = any(not os.path.exists(_out_path(cfg, n))
                              for _, n in CML_WINDOWS)
            if not _exists(cfg, rain_path, 'rain'):
                wanted.append(('rain', rain_path, _first_year(cfg)))
            if (cfg.years_out is None and cml_missing
                    and not _exists(cfg, ext_path, 'rain_extrayear')):
                wanted.append(('extrayear', ext_path, None))
        else:
            path = _out_path(cfg, var)
            if not _exists(cfg, path, var):
                wanted.append((var, path, None))
        if not wanted:
            continue

        if not _nc_files(os.path.join(cfg.interim_path, var)):
            print(f'  WARNING: no files for {var}, skipping')
            continue

        if cfg.verbose:
            print(f'  {var}')

        src = _GapfillSource(cfg, var)
        store = _gapfill_clim_store(cfg, var, src)
        if store.available:
            src.clim = store
        times = src.times
        clim_years = _clim_years(times)

        masks = {k: cfg.out_mask(times, start) for k, _, start in wanted}
        union = np.zeros(len(times), dtype=bool)
        for m in masks.values():
            union |= m
        union_idx = np.where(union)[0]
        # position of each output's time steps within the union
        sub = {k: np.searchsorted(union_idx, np.where(m)[0])
               for k, m in masks.items()}

        tile_r = _tile_rows(len(times), src.n_lon)
        ok = False
        try:
            with _Outputs() as outs:
                for k, path, _ in wanted:
                    outs.add(k, path, src.name, src.lat, src.lon,
                             times[masks[k]].values, cfg.years_out_str)

                def compute(lat_s, lat_e):
                    filled, fresh = src.compute(lat_s, lat_e, union_idx)
                    vals = filled.values
                    return ({k: vals[sub[k]] for k in sub}, fresh)

                def write(lat_s, lat_e, res):
                    arrs, fresh = res
                    for k, a in arrs.items():
                        outs[k].write(lat_s, lat_e, a)
                    if fresh is not None:
                        store.write(lat_s, lat_e, fresh, clim_years)

                _run_tiles(cfg, src.n_lat, tile_r, compute, write, var)
            ok = True
        finally:
            store.finish(ok)
            with _IO_LOCK:
                src.ds.close()
        gc.collect()


# ---------------------------------------------------------------------------
# Step 2b: Vegetation fractions (Donohue et al. 2009)
# ---------------------------------------------------------------------------

def _vegetation_fractions(cfg, ndvi_max=0.91):
    """
    Decompose gap-filled NDVI into trees (persistent), grass (recurrent),
    and bare fractions.  Annual maxima are broadcast to monthly time steps.
    Rolling windows always see the full NDVI history.
    """
    ndvi_min_path = os.path.join(cfg.static_path,
                                 f'ndvi_of_baresoil_{TARGET_GRID}.nc')
    if not os.path.exists(ndvi_min_path):
        print(f'  WARNING: Bare-soil NDVI map not found ({ndvi_min_path}), '
              f'skipping veg fractions')
        return
    ndvi_path = _out_path(cfg, 'NDVI')
    if not os.path.exists(ndvi_path) and \
            not os.path.isdir(os.path.join(cfg.interim_path, 'NDVI')):
        print(f'  WARNING: Gap-filled NDVI not found ({ndvi_path}), '
              f'skipping veg fractions')
        return

    out_paths = {v: _out_path(cfg, v) for v in ('trees', 'grass', 'bare')}
    if all(os.path.exists(p) for p in out_paths.values()):
        if cfg.verbose:
            print('  trees/grass/bare: exist, skipping')
        return

    if cfg.verbose:
        print(f'  Computing vegetation fractions '
              f'(bare {"clipped at 0" if cfg.bare_clip else "unclipped"})')

    src = _feature_source(cfg, 'NDVI', need_full=True)

    with _IO_LOCK:
        ndvi_min_ds = xr.open_dataset(ndvi_min_path)
        ndvi_min_var = _data_var(ndvi_min_ds)
        ndvi_min_2d = ndvi_min_ds[ndvi_min_var].values.astype('float32')
        ndvi_min_ds.close()

    times = src.times
    n_lat, n_lon = src.n_lat, src.n_lon
    years = sorted(set(times.year))
    out_years = [y for y in years
                 if cfg.years_out is None
                 or cfg.years_out[0] <= y <= cfg.years_out[1]]
    monthly_times = [pd.Timestamp(f'{y}-{m:02d}-15')
                     for y in out_years for m in range(1, 13)]

    # Smaller tiles: veg fractions create several intermediate arrays
    tile_r = _tile_rows(len(times), n_lon, max_mem=1.2e9)

    def compute(lat_s, lat_e):
        ndvi_tile = src.read(lat_s, lat_e)
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

        # Annual max; bare is the residual (reference: unclipped)
        persist_ann   = persist.resample(time='YE').max()
        recurrent_ann = recurrent.resample(time='YE').max()
        bare_ann      = 1 - (persist_ann + recurrent_ann)
        if cfg.bare_clip:
            bare_ann = bare_ann.clip(min=0)
        del persist, recurrent

        # Broadcast annual → monthly (constant within each year)
        shape = (len(monthly_times), lat_e - lat_s, n_lon)
        trees_arr = np.empty(shape, dtype='float32')
        grass_arr = np.empty_like(trees_arr)
        bare_arr  = np.empty_like(trees_arr)

        out_idx = 0
        for y in out_years:
            ys = str(y)
            p_slice = persist_ann.sel(time=ys).values.squeeze()
            r_slice = recurrent_ann.sel(time=ys).values.squeeze()
            b_slice = bare_ann.sel(time=ys).values.squeeze()
            for _ in range(12):
                trees_arr[out_idx] = p_slice
                grass_arr[out_idx] = r_slice
                bare_arr[out_idx]  = b_slice
                out_idx += 1
        return trees_arr, grass_arr, bare_arr

    try:
        with _Outputs() as outs:
            for vname in ('trees', 'grass', 'bare'):
                outs.add(vname, out_paths[vname], vname, src.lat, src.lon,
                         monthly_times, cfg.years_out_str)

            def write(lat_s, lat_e, res):
                for vname, arr in zip(('trees', 'grass', 'bare'), res):
                    outs[vname].write(lat_s, lat_e, arr)

            _run_tiles(cfg, n_lat, tile_r, compute, write, 'vegfrac')
    finally:
        src.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2c: Cumulative rainfall
# ---------------------------------------------------------------------------

def _cumulative_rainfall(cfg):
    """
    Compute 3-, 6-, and 12-month rolling cumulative rainfall from the full
    gap-filled rain history (the extrayear file, or inline from interim).
    Remove the extrayear file when done.
    """
    extrayear_path = os.path.join(cfg.results_path,
                                  f'rain_{TARGET_GRID}_extrayear.nc')

    windows_todo = [(w, n) for w, n in CML_WINDOWS
                    if not _exists(cfg, _out_path(cfg, n), n)]

    if not windows_todo:
        if os.path.exists(extrayear_path):
            if cfg.verbose:
                print('  Removing rain extrayear file')
            os.remove(extrayear_path)
        return

    if not (os.path.exists(extrayear_path)
            or os.path.isdir(os.path.join(cfg.interim_path, 'rain'))):
        print('  WARNING: rain extrayear file / rain interim not found, '
              'skipping')
        return

    src = _rain_hist_source(cfg)
    times = src.times
    out_idx = np.where(cfg.out_mask(times, start=CLIM_START))[0]
    out_times = times[out_idx].values
    tile_r = _tile_rows(len(times), src.n_lon)

    if cfg.verbose:
        for _, out_var in windows_todo:
            print(f'  {out_var}')

    def compute(lat_s, lat_e):
        # Load tile eagerly — all time steps, this lat band
        tile = src.read(lat_s, lat_e)
        res = {}
        for window, out_var in windows_todo:
            cml = tile.rolling(time=window, min_periods=window).sum()
            res[out_var] = cml.isel(time=out_idx).values.astype('float32')
            del cml
        return res

    try:
        with _Outputs() as outs:
            for _, out_var in windows_todo:
                outs.add(out_var, _out_path(cfg, out_var), out_var,
                         src.lat, src.lon, out_times, cfg.years_out_str)

            def write(lat_s, lat_e, res):
                for k, a in res.items():
                    outs[k].write(lat_s, lat_e, a)

            _run_tiles(cfg, src.n_lat, tile_r, compute, write, 'cmlrain')
    finally:
        src.close()

    # Remove extrayear file now that cumulative products are written
    if os.path.exists(extrayear_path):
        if cfg.verbose:
            print('  Removing rain extrayear file')
        os.remove(extrayear_path)
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2d: ΔT = LST − Tavg
# ---------------------------------------------------------------------------

def _delta_t(cfg):
    """Compute land surface temperature minus air temperature, per month."""
    out_path = _out_path(cfg, 'ΔT')
    if _exists(cfg, out_path, 'ΔT'):
        return

    for var in ('LST', 'Tavg'):
        if not (os.path.exists(_out_path(cfg, var))
                or os.path.isdir(os.path.join(cfg.interim_path, var))):
            print(f'  WARNING: {var} not found ({_out_path(cfg, var)}), '
                  f'skipping ΔT')
            return

    if cfg.verbose:
        print('  ΔT')

    lst = _feature_source(cfg, 'LST', need_full=False)
    tavg = _feature_source(cfg, 'Tavg', need_full=False)
    try:
        if len(lst.times) != len(tavg.times):
            raise ValueError(f'LST ({len(lst.times)}) and Tavg '
                             f'({len(tavg.times)}) time axes differ')
        out_idx = np.where(cfg.out_mask(lst.times))[0]
        tile_r = _tile_rows(len(lst.times), lst.n_lon)

        def compute(lat_s, lat_e):
            a = lst.read(lat_s, lat_e, out_idx).values
            b = tavg.read(lat_s, lat_e, out_idx).values
            return a - b

        with _Outputs() as outs:
            outs.add('dt', out_path, 'ΔT', lst.lat, lst.lon,
                     lst.times[out_idx].values, cfg.years_out_str)
            _run_tiles(cfg, lst.n_lat, tile_r, compute,
                       lambda s, e, r: outs['dt'].write(s, e, r), 'ΔT')
    finally:
        lst.close()
        tavg.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Step 2e: Fractional anomalies
# ---------------------------------------------------------------------------

def _fractional_anomalies(cfg):
    """
    Compute fractional anomalies: variable / monthly_climatology(2003–2022).

    anom_guard 'one': where the source is finite but the anomaly is not
    (0/0, x/0), set the anomaly to 1.0.  'none': leave NaN/Inf (reference).
    """
    anom_vars = [v for v in ANOM_VARS if cfg.vars is None or v in cfg.vars]
    if not anom_vars:
        print('  no anomaly variables selected')
        return

    for var in anom_vars:
        anom_var = f'{var}_anom'
        out_path = _out_path(cfg, anom_var)
        if _exists(cfg, out_path, anom_var):
            continue

        # Need the full input history only if the climatology is not stored.
        in_path = _out_path(cfg, var)
        store = None
        try:
            src = _anom_input_source(cfg, var, need_full=False)
        except (FileNotFoundError, StopIteration) as e:
            print(f'  WARNING: {var} not available ({e}), skipping {anom_var}')
            continue
        store = _ClimStore(cfg, var, src.lat, src.lon)
        if not store.available and isinstance(src, _FileSource) \
                and not _usable(cfg, in_path, need_full=True):
            src.close()
            src = _anom_input_source(cfg, var, need_full=True)

        if cfg.verbose:
            print(f'  {anom_var}')

        times = src.times
        out_idx = np.where(cfg.out_mask(times, start=CLIM_START))[0]
        tile_r = _tile_rows(len(times), src.n_lon)
        clim_years = _clim_years(times)

        def compute(lat_s, lat_e):
            if store.available:
                tile = src.read(lat_s, lat_e, out_idx)
                clim = store.read(lat_s, lat_e, tile)
                fresh = None
                anom = tile.groupby('time.month') / clim
                anom = anom.drop_vars('month')
                src_out = tile
            else:
                tile = src.read(lat_s, lat_e)
                clim = _clim_subset(tile, f'{var} (anom)') \
                           .groupby('time.month').mean('time')
                fresh = clim
                anom = tile.groupby('time.month') / clim
                anom = anom.drop_vars('month').isel(time=out_idx)
                src_out = tile.isel(time=out_idx)

            if cfg.anom_guard == 'one':
                # Guard the fractional anomaly against 0/0 (NaN) and x/0
                # (Inf), which occur in e.g. zero-rainfall months where the
                # 2003-2022 climatology is also zero.  Where the source value
                # is finite but the anomaly is not, set it to 1.0 (== at
                # climatological level) instead of leaving NaN/Inf.
                anom = anom.where(
                    ~(np.isfinite(src_out) & ~np.isfinite(anom)), 1.0)
            return anom.values.astype('float32'), fresh

        ok = False
        try:
            with _Outputs() as outs:
                outs.add('a', out_path, anom_var, src.lat, src.lon,
                         times[out_idx].values, cfg.years_out_str)

                def write(lat_s, lat_e, res):
                    arr, fresh = res
                    outs['a'].write(lat_s, lat_e, arr)
                    if fresh is not None:
                        store.write(lat_s, lat_e, fresh, clim_years)

                _run_tiles(cfg, src.n_lat, tile_r, compute, write, anom_var)
            ok = True
        finally:
            store.finish(ok)
            src.close()
        gc.collect()


# ---------------------------------------------------------------------------
# Step 2f: C4 grass fraction
# ---------------------------------------------------------------------------

def _c4_grass_fraction(cfg):
    """
    Combine a static C4 grass coverage map with the dynamic grass fraction
    to produce a time-varying C4 grass fraction.
    """
    c4_path = os.path.join(cfg.static_path, 'Aust_C4_grass_cover_percentage.tif')
    grass_path = _out_path(cfg, 'grass')
    mask_path  = os.path.join(cfg.static_path, f'land_sea_mask_{TARGET_GRID}.nc')
    out_path   = _out_path(cfg, 'C4_grass')

    if _exists(cfg, out_path, 'C4_grass'):
        return

    for p, label in [(c4_path, 'C4 grass TIF'),
                     (grass_path, 'Grass fraction'),
                     (mask_path, 'Land mask')]:
        if not os.path.exists(p):
            print(f'  WARNING: {label} not found ({p}), skipping C4 grass')
            return
    if not _usable(cfg, grass_path, need_full=False):
        print(f'  WARNING: {grass_path} was written with years_out='
              f'{_file_years_out(grass_path)}, requested '
              f'{cfg.years_out_str}; rerun vegfrac. Skipping C4 grass')
        return

    if cfg.verbose:
        print('  C4_grass')

    # Load and reproject C4 coverage to target grid
    grass = _FileSource(grass_path)
    dst_lat, dst_lon = grass.lat, grass.lon

    with _IO_LOCK:
        mask_ds = xr.open_dataset(mask_path)
        mask_var = _data_var(mask_ds)
        mask_arr = mask_ds[mask_var].values
        mask_ds.close()

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
    out_idx = np.where(cfg.out_mask(grass.times))[0]
    try:
        with _Outputs() as outs:
            f = outs.add('c4', out_path, 'C4_grass', dst_lat, dst_lon,
                         grass.times[out_idx].values, cfg.years_out_str)
            for i, t in enumerate(out_idx):
                with _IO_LOCK:
                    grass_slice = grass.da.isel(time=int(t)).values \
                                  .astype('float32')
                f.write(0, len(dst_lat), grass_slice * c4_reprojected, t=i)
                del grass_slice
    finally:
        grass.close()
    gc.collect()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

_STAGE_FUNCS = {
    'gapfill': ('Step 2a: Combining datasets + gap-filling', _combine_gapfill),
    'vegfrac': ('Step 2b: Vegetation fractions', _vegetation_fractions),
    'cmlrain': ('Step 2c: Cumulative rainfall', _cumulative_rainfall),
    'deltat':  ('Step 2d: LST minus Tair (ΔT)', _delta_t),
    'anom':    ('Step 2e: Fractional anomalies', _fractional_anomalies),
    'c4':      ('Step 2f: C4 grass fraction', _c4_grass_fraction),
}


def create_feature_datasets(interim_path=None, results_path=None,
                            static_path=None, verbose=False, stages=None,
                            bottleneck=False, flox=False, tile_rows=None,
                            **kwargs):
    """
    Run the Step 2 pipeline (all stages by default, in this order):
      2a  gapfill  Combine + gap-fill
      2b  vegfrac  Vegetation fractions
      2c  cmlrain  Cumulative rainfall
      2d  deltat   ΔT
      2e  anom     Fractional anomalies
      2f  c4       C4 grass fraction
    Extra keyword arguments are passed to Config (years_out, bare_clip,
    anom_guard, vars_, reuse_clim, threads, pool).  *bottleneck* / *flox* enable
    xarray's accelerated rolling / groupby paths (off by default: they change
    results at the float32 rounding level, see --help).
    """
    cfg = Config(interim_path or DEFAULT_INTERIM,
                 results_path or DEFAULT_RESULTS,
                 static_path or DEFAULT_STATIC,
                 verbose=verbose, **kwargs)
    os.makedirs(cfg.results_path, exist_ok=True)
    global _TILE_ROWS_OVERRIDE
    _TILE_ROWS_OVERRIDE = tile_rows

    stages = STAGES if stages is None else [s for s in STAGES if s in stages]
    if verbose:
        print(f'interim={cfg.interim_path}\nresults={cfg.results_path}\n'
              f'static={cfg.static_path}\nstages={",".join(stages)} '
              f'vars={"all" if cfg.vars is None else ",".join(sorted(cfg.vars))} '
              f'years_out={cfg.years_out_str or "all"} '
              f'bare_clip={cfg.bare_clip} anom_guard={cfg.anom_guard} '
              f'reuse_clim={cfg.reuse_clim} threads={cfg.threads} ({cfg.pool}) '
              f'bottleneck={bottleneck} flox={flox}')

    with xr.set_options(use_bottleneck=bottleneck, use_flox=flox):
        for st in stages:
            title, func = _STAGE_FUNCS[st]
            if verbose:
                print(title)
            t0 = time.time()
            func(cfg)
            if verbose:
                print(f'  [{st}: {time.time() - t0:.1f} s, writes {_WT[0]:.1f} s]'); _WT[0] = 0

    if verbose:
        print('Step 2 complete.')
        print(f'Outputs written to: {cfg.results_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_years(s):
    try:
        a, b = (int(x) for x in s.split('-'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected YYYY-YYYY, got {s!r}')
    if a > b:
        raise argparse.ArgumentTypeError(f'empty year range {s!r}')
    return a, b


def _parse_list(s):
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 2: Feature dataset construction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""stages and their inputs (run in this order):
  gapfill  interim/{var}/{var}_500m_{year}.nc -> {var}_500m.nc
           (rain -> rain_500m.nc + rain_500m_extrayear.nc)
  vegfrac  needs gap-filled NDVI (+ static ndvi_of_baresoil) -> trees, grass, bare
  cmlrain  needs gap-filled rain incl. the prior year -> rain_cml3/6/12
  deltat   needs gap-filled LST and Tavg -> ΔT
  anom     needs gap-filled NDWI SRAD Tavg VPD kNDVI LAI, rain and
           rain_cml* -> *_anom (only the climatology if already stored)
  c4       needs grass (from vegfrac) + static C4 tif + land mask -> C4_grass
A missing (or --years_out-truncated) gap-filled input is recomputed inline
from the interim files, so stages may run in separate jobs in any grouping
as long as vegfrac precedes c4.  Existing outputs are never overwritten.

environment: AUSEFLUX_ROOT, AUSEFLUX_INPUT, AUSEFLUX_STATIC set the defaults.
""")
    parser.add_argument('--interim_path', type=str, default=DEFAULT_INTERIM,
                        help='Step 1 output directory. Default: '
                             '$AUSEFLUX_ROOT/interim_500m '
                             f'({DEFAULT_INTERIM})')
    parser.add_argument('--results_path', type=str, default=DEFAULT_RESULTS,
                        help='Output directory. Default: '
                             '$AUSEFLUX_ROOT/features_500m '
                             f'({DEFAULT_RESULTS})')
    parser.add_argument('--static_path', type=str, default=DEFAULT_STATIC,
                        help='Static input files. Default: $AUSEFLUX_STATIC '
                             f'({DEFAULT_STATIC})')
    parser.add_argument('--stages', type=_parse_list, default=None,
                        help='Comma-separated subset of '
                             f'{",".join(STAGES)} (default: all, in that '
                             'order). See the dependency notes below.')
    parser.add_argument('--vars', type=_parse_list, default=None,
                        help='Comma-separated variables for the gapfill stage '
                             '(interim variable names, e.g. NDWI,rain) and the '
                             'anom stage (anomaly inputs: '
                             f'{",".join(ANOM_VARS)}). Default: all. Other '
                             'stages ignore it. For PBS array sharding.')
    parser.add_argument('--years_out', type=_parse_years, default=None,
                        metavar='YYYY-YYYY',
                        help='Write only these years to every output file '
                             '(computation still uses the full input history: '
                             'climatologies, rolling windows, interpolation). '
                             'Outputs are bit-identical to the matching slice '
                             'of a full run. Default: all years.')
    parser.add_argument('--bare_clip', dest='bare_clip', action='store_true',
                        default=True,
                        help='Clip the annual bare fraction at 0 (default).')
    parser.add_argument('--no_bare_clip', dest='bare_clip',
                        action='store_false',
                        help='Reference behaviour: bare = 1 - (trees + grass) '
                             'without a lower clip (negative where the '
                             'annual maxima of trees and grass sum above 1).')
    parser.add_argument('--anom_guard', choices=['one', 'none'], default='one',
                        help="Non-finite fractional anomalies (0/0, x/0) where "
                             "the source is finite: 'one' sets them to 1.0 "
                             "(default); 'none' leaves NaN/Inf as the "
                             "reference does.")
    parser.add_argument('--reuse_clim', dest='reuse_clim', action='store_true',
                        default=True,
                        help='Reuse monthly climatologies stored in '
                             '{results_path}/climatology/ (default); results '
                             'are bit-identical to recomputing them.')
    parser.add_argument('--no_reuse_clim', dest='reuse_clim',
                        action='store_false',
                        help='Recompute climatologies (and overwrite the '
                             'stored ones). Use if the 2003-2022 inputs '
                             'changed.')
    parser.add_argument('--threads', type=int, default=1,
                        help='Latitude tiles computed concurrently in the '
                             'tiled stages (gapfill, vegfrac, cmlrain, deltat, '
                             'anom); at most N tiles in flight, so memory '
                             'scales with N. Writes stay in the main process. '
                             'Results are bit-identical. Default: 1.')
    parser.add_argument('--pool', choices=['process', 'thread'],
                        default='process',
                        help="Worker type for --threads > 1: 'process' "
                             "(forked processes, default; real speed-up) or "
                             "'thread' (xarray groupby/interpolation hold the "
                             "GIL, so little gain).")
    parser.add_argument('--tile_rows', type=int, default=None,
                        help='Latitude rows per tile (default: automatic, '
                             '50-400 rows for ~2 GB per tile). Results do not '
                             'depend on it.')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                        help='Use bottleneck for rolling min/mean/sum (~8x '
                             'faster rolling windows in vegfrac/cmlrain). Off '
                             'by default: its running sums differ from the '
                             'default path at the float32 rounding level '
                             '(max rel. diff ~1e-5).')
    parser.add_argument('--flox', action='store_true', default=False,
                        help='Use flox for groupby(month) reductions. Off by '
                             'default: changes climatologies at the float32 '
                             'rounding level and was not faster on '
                             'in-memory tiles.')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    if args.stages is not None:
        bad = [s for s in args.stages if s not in STAGES]
        if bad:
            parser.error(f'unknown stage(s) {bad}; choose from {STAGES}')
    vars_ = None
    if args.vars is not None:
        known = set(ANOM_VARS)
        if os.path.isdir(args.interim_path):
            known |= set(_feature_vars(args.interim_path))
        bad = [v for v in args.vars if v not in known]
        if bad:
            parser.error(f'unknown variable(s) {bad}; known: {sorted(known)}')
        vars_ = set(args.vars)
    if args.threads < 1:
        parser.error('--threads must be >= 1')

    create_feature_datasets(
        interim_path=args.interim_path,
        results_path=args.results_path,
        static_path=args.static_path,
        verbose=args.verbose,
        stages=args.stages,
        years_out=args.years_out,
        bare_clip=args.bare_clip,
        anom_guard=args.anom_guard,
        vars_=vars_,
        reuse_clim=args.reuse_clim,
        threads=args.threads,
        pool=args.pool,
        bottleneck=args.bottleneck,
        flox=args.flox,
        tile_rows=args.tile_rows,
    )


if __name__ == '__main__':
    main()
