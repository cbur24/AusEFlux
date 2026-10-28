"""
Step 3 (Quantile): Train quantile LightGBM models and predict AusEFlux.

Instead of running 30 ensemble models and combining empirical quantiles,
this script trains 3 quantile-regression LightGBM models per flux
(alpha=0.25, 0.50, 0.75) and predicts directly.  The result is the same
output format as step3+step4 combined, but ~10x faster at inference.

Training mode  (--mode train):
    Collects site-level training CSVs and trains 3 quantile models per flux.

Predict mode   (--mode predict, default):
    Applies the 3 pre-trained quantile models to produce the final output.

Paths default to $AUSEFLUX_ROOT (default /home/prl900/Downloads/wald_eflux):
features {root}/features_500m, static $AUSEFLUX_STATIC ({root}/input/static),
results {root}/results (variables.txt, models/quantile/v2.1/{flux},
AusEFlux_quantile/{flux}); each is overridable on the command line.
Predict-mode switches: --mask_vars, --mask_test, --valid_only/--all_pixels,
--threads, --months/--merge/--merge_partial, --lat_rows START:END (same
part-file / _r{START}-{END} window naming, feature-window detection, input
validation, configuration attributes (auseflux_mask_vars, auseflux_mask_test,
auseflux_valid_only; windowed outputs add auseflux_lat_rows) and non-zero exit when an output was not produced as
step3_ensemble_prediction.py).  The no-data
defaults follow the reference ensemble (and step3_ensemble_prediction.py):
mask vars LAI_anom,VegH,NDWI,rain_anom,Tavg, NaN-only test (`isnull()`;
+-Inf is predicted with Inf -> 0), valid_only on (bit-identical, faster).

Usage:
    uv run step3_ensemble_prediction_quantile.py --year 2024 --mode train --training_data data/training_csvs
    uv run step3_ensemble_prediction_quantile.py --year 2024 --mode predict
    uv run step3_ensemble_prediction_quantile.py --year 2024 --fluxes GPP ET
    uv run step3_ensemble_prediction_quantile.py --year 2024 --fluxes GPP --months 1-6
    uv run step3_ensemble_prediction_quantile.py --year 2024 --fluxes GPP --merge
"""

# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "numpy<2",
#     "pandas",
#     "xarray>=2025.1.0",
#     "netCDF4",
#     "joblib",
#     "lightgbm",
#     "scikit-learn==1.3.2",
# ]
# ///
# NOTE: scikit-learn is pinned to 1.3.2 (and numpy<2) to match the version the
# reference ensemble RandomForest models were trained with.  Newer scikit-learn
# (>=1.4) fails to unpickle those models (missing 'monotonic_cst' attribute).

import argparse
import gc
import glob
import os
import re
import resource
import sys
import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')

import joblib
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _env(name, default):
    """Environment variable *name* if set and non-empty, else *default*."""
    v = os.environ.get(name)
    return v if v else default


BASE_PATH = _env('AUSEFLUX_ROOT', '/home/prl900/Downloads/wald_eflux')
INPUT_PATH = _env('AUSEFLUX_INPUT', os.path.join(BASE_PATH, 'input'))
DEFAULT_FEATURES = os.path.join(BASE_PATH, 'features_500m')
DEFAULT_STATIC   = _env('AUSEFLUX_STATIC', os.path.join(INPUT_PATH, 'static'))
DEFAULT_RESULTS  = os.path.join(BASE_PATH, 'results')
TARGET_GRID = '500m'
VERSION = 'v2.1'

# Default no-data masking variables = the reference
# (master:src/_ensemble_prediction.py).  The pre-2026-10 refactor default
# omitted 'LAI_anom' (--mask_vars VegH,NDWI,rain_anom,Tavg).
MASKING_VARS = ['LAI_anom', 'VegH', 'NDWI', 'rain_anom', 'Tavg']

FLUX_METADATA = {
    'GPP': {
        'long_name': 'Gross Primary Productivity',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
        'target_col': 'GPP_SOLO_EC',
    },
    'NEE': {
        'long_name': 'Net Ecosystem Exchange',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
        'target_col': 'NEE_SOLO_EC',
    },
    'ER': {
        'long_name': 'Ecosystem Respiration',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
        'target_col': 'ER_SOLO_EC',
    },
    'ET': {
        'long_name': 'Evapotranspiration',
        'units': 'mm/month',
        'target_col': 'ET_EC',
    },
}

QUANTILES = {
    'q25': 0.25,
    'q50': 0.50,
    'q75': 0.75,
}

TILE_ROWS = 1400  # latitude rows per tile (multiple of the 200-row output chunks)
PREDICT_N_JOBS = -1  # LightGBM predict threads (-1 = all cores)

# Feature column order (must match variables.txt / original model training)
FEATURE_COLS = [
    'LST_RS', 'ΔT_RS', 'LAI_RS', 'LAI_anom_RS', 'kNDVI_RS',
    'kNDVI_anom_RS', 'NDWI_RS', 'NDWI_anom_RS', 'trees_RS', 'grass_RS',
    'bare_RS', 'C4_grass_RS', 'rain_RS', 'rain_cml3_RS', 'rain_anom_RS',
    'rain_cml3_anom_RS', 'rain_cml6_anom_RS', 'rain_cml12_anom_RS',
    'SRAD_RS', 'SRAD_anom_RS', 'Tavg_RS', 'Tavg_anom_RS', 'VPD_RS',
    'VPD_anom_RS', 'VegH_EC',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_var(ds):
    """Return first non-metadata data variable name."""
    skip = {'crs', 'spatial_ref'}
    return next(v for v in ds.data_vars if v not in skip and ds[v].ndim > 0)


def _coords(da):
    """Return (lat, lon) float32 arrays rounded to 4 d.p."""
    lat = np.array([round(float(v), 4) for v in da.latitude.values],
                   dtype='float32')
    lon = np.array([round(float(v), 4) for v in da.longitude.values],
                   dtype='float32')
    return lat, lon


def _parse_variables(features_list):
    """Parse variables.txt into an ordered list of feature variable names."""
    with open(features_list) as f:
        header = f.read().strip().split(',')
    train_vars = [v for v in header if v and v != 'site']
    train_vars = [v[:-3] for v in train_vars]
    return train_vars


def _parse_months(spec):
    """Parse '1-12', '1,6,12' or '1-3,7' into a sorted list of month ints."""
    months = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = (int(x) for x in part.split('-', 1))
            months.update(range(a, b + 1))
        else:
            months.add(int(part))
    if not months or min(months) < 1 or max(months) > 12:
        raise ValueError(f'invalid --months specification: {spec!r}')
    return sorted(months)


def _parse_lat_rows(spec):
    """Parse 'START:END' (0-based rows, END exclusive) into (start, end)."""
    try:
        a, b = (int(x) for x in str(spec).split(':'))
    except ValueError:
        raise ValueError(f'invalid --lat_rows {spec!r} (expected START:END)')
    if a < 0 or b <= a:
        raise ValueError(f'invalid --lat_rows {spec!r} (need 0 <= START < END)')
    return a, b


def _rows_suffix(lat_rows):
    """File-name suffix for a latitude window ('' for the full grid)."""
    return f'_r{lat_rows[0]}-{lat_rows[1]}' if lat_rows else ''


def _rows_attr(lat_rows):
    """Value of the `auseflux_lat_rows` global attribute (None = full)."""
    return f'{lat_rows[0]}:{lat_rows[1]}' if lat_rows else None


def _product_name(flux, year, lat_rows=None):
    return (f'AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year}_{VERSION}'
            f'{_rows_suffix(lat_rows)}.nc')


def _part_name(flux, year, month, lat_rows=None):
    return (f'AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year}_{VERSION}'
            f'_m{month:02d}{_rows_suffix(lat_rows)}.nc')


def _peak_rss_gb():
    # ru_maxrss is in KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


LAT_ROWS_ATTR = 'auseflux_lat_rows'

# Run-configuration global attributes written on products and month parts.  Month parts must agree on all of them to be merged, and an
# existing output whose values differ from the request is reported (WARNING)
# when it is skipped.
# (quantile_method / n_models of the ensemble script do not apply here: the
# bands come from 3 quantile-regression models, see the 'method' attribute.)
CONFIG_ATTRS = ('auseflux_mask_vars', 'auseflux_mask_test',
                'auseflux_valid_only')


def _config_attrs(mask_vars, mask_test, valid_only):
    """Run-configuration attributes (see CONFIG_ATTRS; `auseflux_` prefix so
    they sit beside the published v2.1 attributes).  *mask_vars* are the
    effective mask variables.  The latitude window is a separate attribute
    (LAT_ROWS_ATTR, windowed outputs only) and part of the file name."""
    return {
        'auseflux_mask_vars': ','.join(mask_vars),
        'auseflux_mask_test': mask_test,
        'auseflux_valid_only': int(bool(valid_only)),
    }


def _file_attrs(path):
    with nc4.Dataset(path) as d:
        return {k: d.getncattr(k) for k in d.ncattrs()}


def _config_diff(attrs, config):
    """'key: file=X requested=Y' for each configuration attribute that
    differs between a file's global *attrs* and the requested *config*."""
    out = []
    for k, v in config.items():
        have = attrs.get(k, '(absent)')
        if str(have) != str(v):
            out.append(f'{k}: file={have} requested={v}')
    return out


def _report_existing(path, config):
    """Skip message for an existing output (+ WARNING if its configuration
    attributes differ from the requested *config*)."""
    print(f'  exists, skipping: {path}')
    try:
        diff = _config_diff(_file_attrs(path), config)
    except Exception as e:  # unreadable file: report, keep skipping
        diff = [f'(could not read its attributes: {e!r})']
    if diff:
        print(f'  WARNING: {os.path.basename(path)} was produced with a '
              f'different configuration than requested (kept as is; delete '
              f'it to regenerate):')
        for d in diff:
            print(f'    {d}')


def _fmt_times(values, n=3):
    t = [f'{pd.Timestamp(v):%Y-%m-%d}' for v in values]
    if len(t) > 2 * n + 1:
        t = t[:n] + ['...'] + t[-n:]
    return f'{len(values)} steps [{", ".join(t)}]'


def _coords_close(a, b, tol=1e-4):
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    return a.shape == b.shape and (a.size == 0 or np.abs(a - b).max() <= tol)


def _open_features(features_path, train_vars, year, timestamps, lat, lon,
                   feat_slice=None, feat_rows=None):
    """
    Open every feature lazily for *year* (rows *feat_slice* of the file, if
    given) and validate it against the reference feature (the first of
    *train_vars*): same `auseflux_lat_rows` attribute (*feat_rows*; None =
    full grid), identical time axis (*timestamps*) and latitude/longitude
    within 1e-4 deg of *lat*/*lon*.  Features are read by time POSITION and
    row offset later, so any mismatch is an error.
    Returns ({var: DataArray}, None) or (None, error message).
    """
    exp_t = np.array([pd.Timestamp(t) for t in timestamps],
                     dtype='datetime64[ns]')
    out = {}

    def fail(ds, msg):
        if ds is not None:
            ds.close()
        for da in out.values():
            da.close()
        return None, msg

    for var in train_vars:
        path = os.path.join(features_path, f'{var}_{TARGET_GRID}.nc')
        if not os.path.exists(path):
            return fail(None, f'feature file not found: {path}')
        ds = xr.open_dataset(path)
        rows = ds.attrs.get(LAT_ROWS_ATTR)
        if rows != feat_rows:
            return fail(ds, f'feature {var} ({path}) covers latitude rows '
                            f'{rows or "(full grid)"} but {train_vars[0]} '
                            f'covers {feat_rows or "(full grid)"}')
        try:
            da = ds[_data_var(ds)].sel(time=str(year))
        except KeyError:
            return fail(ds, f'feature {var} ({path}) has no time steps in '
                            f'{year}')
        tv = np.asarray(da.time.values, dtype='datetime64[ns]')
        if not np.array_equal(tv, exp_t):
            return fail(ds, f'time axis of feature {var} ({path}) in {year} '
                            f'differs from that of {train_vars[0]}: '
                            f'{_fmt_times(tv)} vs {_fmt_times(exp_t)}')
        if feat_slice is not None:  # lazy: only the window rows are read
            da = da.isel(latitude=slice(*feat_slice))
        if not (_coords_close(da.latitude.values, lat)
                and _coords_close(da.longitude.values, lon)):
            return fail(ds, f'latitude/longitude of feature {var} ({path}) '
                            f'differ from those of {train_vars[0]}')
        out[var] = da
    return out, None


def _load_urban_mask(static_path, out_rows, lat, lon):
    """
    Urban/water mask (True = masked) for full-grid rows *out_rows* (None =
    all), checked against the feature grid *lat*/*lon* (1e-4 deg).  Exits
    with an error on mismatch; all-False (with a WARNING) if the file is
    absent.
    """
    urban_path = os.path.join(static_path,
                              f'urban_water_mask_{TARGET_GRID}.nc')
    if not os.path.exists(urban_path):
        print(f'  WARNING: urban/water mask not found ({urban_path}), '
              f'proceeding without it')
        return np.zeros((len(lat), len(lon)), dtype=bool)
    with xr.open_dataset(urban_path) as urban_ds:
        urban_da = urban_ds[_data_var(urban_ds)]
        n_full = urban_da.sizes['latitude']
        if out_rows is not None:
            if out_rows[1] > n_full:
                print(f'ERROR: latitude rows {_rows_attr(out_rows)} exceed the '
                      f'urban/water mask grid ({n_full} rows, {urban_path})')
                sys.exit(1)
            urban_da = urban_da.isel(latitude=slice(*out_rows))
        u_lat = urban_da.latitude.values
        u_lon = urban_da.longitude.values
        if not (_coords_close(u_lat, lat) and _coords_close(u_lon, lon)):
            def desc(la, lo):
                return (f'{len(la)} x {len(lo)}, lat {float(la[0]):.4f}..'
                        f'{float(la[-1]):.4f}, lon {float(lo[0]):.4f}..'
                        f'{float(lo[-1]):.4f}')
            print(f'ERROR: the urban/water mask {urban_path} (rows '
                  f'{_rows_attr(out_rows) or "all"}: {desc(u_lat, u_lon)}) '
                  f'does not match the feature grid ({desc(lat, lon)}) within '
                  f'1e-4 deg. Latitude-windowed features must carry the '
                  f'{LAT_ROWS_ATTR} attribute (step2 --lat_rows), and '
                  f'--lat_rows is relative to the full grid.')
            sys.exit(1)
        return urban_da.transpose('latitude', 'longitude').values == 1


def _feature_grid(features_path, ref_var, year_str, lat_rows, verbose=False):
    """
    Grid of the run from the reference feature file *ref_var*.

    Feature files written by step2 --lat_rows carry the global attribute
    `auseflux_lat_rows`='S:E' (the FULL-grid rows they hold).  The output
    window *out_rows* (full-grid rows; names the outputs and slices the
    urban mask) is --lat_rows if given (it must lie inside the feature
    window), else the feature window, else None (full grid).  *feat_slice*
    is the row slice of the feature files to read (None = all rows).

    Returns dict(lat, lon, timestamps, out_rows, feat_slice, feat_rows).
    Exits with an error on an inconsistent request.
    """
    ref_path = os.path.join(features_path, f'{ref_var}_{TARGET_GRID}.nc')
    with xr.open_dataset(ref_path) as ref_ds:
        feat_rows = ref_ds.attrs.get(LAT_ROWS_ATTR)
        try:
            ref_da = ref_ds[_data_var(ref_ds)].sel(time=year_str)
        except KeyError:
            print(f'ERROR: {ref_path} has no time steps in {year_str}')
            sys.exit(1)
        n_file = ref_da.sizes['latitude']
        feat_window = None
        if feat_rows is not None:
            try:
                feat_window = _parse_lat_rows(feat_rows)
            except ValueError:
                feat_window = None
            if feat_window is None or feat_window[1] - feat_window[0] != n_file:
                print(f'ERROR: {ref_path}: {LAT_ROWS_ATTR}={feat_rows!r} is '
                      f'inconsistent with its {n_file} latitude rows')
                sys.exit(1)
        if lat_rows is not None:
            if feat_window is not None:
                if not (feat_window[0] <= lat_rows[0]
                        and lat_rows[1] <= feat_window[1]):
                    print(f'ERROR: --lat_rows {_rows_attr(lat_rows)} (rows of '
                          f'the full grid) is not inside the latitude window '
                          f'{feat_rows} of the feature files ({ref_path} '
                          f'{LAT_ROWS_ATTR})')
                    sys.exit(1)
                feat_slice = (lat_rows[0] - feat_window[0],
                              lat_rows[1] - feat_window[0])
            else:
                if lat_rows[1] > n_file:
                    print(f'ERROR: --lat_rows {_rows_attr(lat_rows)} exceeds '
                          f'the feature grid ({n_file} latitude rows)')
                    sys.exit(1)
                feat_slice = lat_rows
            out_rows = lat_rows
        else:
            feat_slice, out_rows = None, feat_window
        if feat_slice is not None:
            ref_da = ref_da.isel(latitude=slice(*feat_slice))
        lat, lon = _coords(ref_da)
        timestamps = [pd.Timestamp(t) for t in ref_da.time.values]
    if verbose:
        src = (f'features hold rows {feat_rows}' if feat_rows
               else 'full-grid features')
        print(f'Grid: {len(lat)} x {len(lon)}, rows '
              f'{_rows_attr(out_rows) or "all"} ({src}), {len(timestamps)} '
              f'time steps for year {year_str}')
    return dict(lat=lat, lon=lon, timestamps=timestamps, out_rows=out_rows,
                feat_slice=feat_slice, feat_rows=feat_rows)


# ---------------------------------------------------------------------------
# NetCDF4 output writer
# ---------------------------------------------------------------------------
# CRS variable of the published v2.1 products (odc-geo `assign_crs` on
# EPSG:4326): an int scalar `spatial_ref` with the WKT2 string, referenced by
# `grid_mapping` on every data variable.
SPATIAL_REF_ATTRS = {
    'spatial_ref': 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
    'crs_wkt': 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
    'semi_major_axis': 6378137.0,
    'semi_minor_axis': 6356752.314245179,
    'inverse_flattening': 298.257223563,
    'reference_ellipsoid_name': 'WGS 84',
    'longitude_of_prime_meridian': 0.0,
    'prime_meridian_name': 'Greenwich',
    'geographic_crs_name': 'WGS 84',
    'horizontal_datum_name': 'World Geodetic System 1984 ensemble',
    'grid_mapping_name': 'latitude_longitude',
}
GRID_MAPPING_VAR = 'spatial_ref'


def _time_units(times):
    """`days since YYYY-01-15` for the year of the first time step: the
    encoding of the published products (xarray's default, int days)."""
    return f'days since {pd.Timestamp(times[0]).year}-01-15'


def _decode_time(var, i=0):
    """Timestamp of time step *i* from a netCDF time variable's units."""
    m = re.match(r'days since (\d{4}-\d{2}-\d{2})', str(getattr(var, 'units', '')))
    if not m:
        raise ValueError(f'unsupported time units {getattr(var, "units", None)!r}')
    return pd.Timestamp(m.group(1)) + pd.Timedelta(days=float(np.asarray(var[:])[i]))


@contextmanager
def _nc_full(path, var_names, lat, lon, times, attrs=None):
    """Create output NetCDF with multiple variables sharing the same grid.

    Layout of the published v2.1 products: int `time` in days since
    15 January of the year, float32 latitude/longitude, a `spatial_ref`
    CRS variable referenced by `grid_mapping`, global `nodata = NaN`; the
    data variables are additionally chunked (1, 200, 200) and deflated.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds = nc4.Dataset(path, 'w', format='NETCDF4')
    try:
        ds.createDimension('time', len(times))
        ds.createDimension('latitude', len(lat))
        ds.createDimension('longitude', len(lon))

        tv = ds.createVariable('time', 'i4', ('time',))
        tv.units = _time_units(times)
        epoch = pd.Timestamp(tv.units[len('days since '):])
        for i, t in enumerate(times):
            delta = pd.Timestamp(t) - epoch
            if delta != pd.Timedelta(days=delta.days):
                raise ValueError(f'time step {t} is not a whole day from {epoch}')
            tv[i] = int(delta.days)

        lv = ds.createVariable('latitude', 'f4', ('latitude',),
                               fill_value=np.float32(np.nan))
        lv[:] = lat

        xv = ds.createVariable('longitude', 'f4', ('longitude',),
                               fill_value=np.float32(np.nan))
        xv[:] = lon

        sr = ds.createVariable(GRID_MAPPING_VAR, 'i4')
        for k, v in SPATIAL_REF_ATTRS.items():
            sr.setncattr(k, v)
        sr.assignValue(0)

        for vn in var_names:
            v = ds.createVariable(vn, 'f4', ('time', 'latitude', 'longitude'),
                                  fill_value=np.float32(np.nan), zlib=True,
                                  complevel=4,
                                  chunksizes=(1, min(200, len(lat)),
                                              min(200, len(lon))))
            v.grid_mapping = GRID_MAPPING_VAR

        ds.setncattr('nodata', np.float64(np.nan))
        if attrs:
            for k, v in attrs.items():
                if k == 'nodata' or (isinstance(v, float) and np.isnan(v)):
                    continue
                ds.setncattr(k, v)

        yield ds
    finally:
        ds.close()


def _merge_parts(final_file, month_parts, year, lat_rows=None, verbose=False):
    """
    Assemble per-month part files into *final_file* with the same variables,
    encoding and global attributes.  *month_parts* is a list of
    (month, path) in month order: all 12 months for --merge, possibly fewer
    for --merge_partial (the time dimension is then the number of parts and
    the global attribute auseflux_partial_months lists them).  Part files are
    left in place.  Returns True on success.

    Every part must hold exactly one time step, of its month of *year*;
    carry the same latitude window as *lat_rows* (`auseflux_lat_rows`
    attribute; absent = full grid: windowed parts are never merged into a
    full product); have the same grid and variables; and the same
    run-configuration attributes (CONFIG_ATTRS).
    """
    part_files = [p for _, p in month_parts]
    missing = [p for p in part_files if not os.path.exists(p)]
    if missing:
        print('  ERROR: missing month part files:')
        for p in missing:
            print(f'    {p}')
        return False
    if not month_parts:
        print('  ERROR: no month part files to merge')
        return False

    timestamps, lat, lon, var_names, attrs = [], None, None, None, None
    cfg0, p0 = None, None
    for m, p in month_parts:
        with nc4.Dataset(p) as src:
            src.set_auto_mask(False)
            tvals = np.asarray(src['time'][:])
            if tvals.size != 1:
                print(f'  ERROR: {p} has {tvals.size} time steps (expected 1)')
                return False
            ts = _decode_time(src['time'])
            if ts.year != int(year) or ts.month != m:
                print(f'  ERROR: {p} holds {ts:%Y-%m}, expected {year}-{m:02d}')
                return False
            p_rows = (src.getncattr(LAT_ROWS_ATTR)
                      if LAT_ROWS_ATTR in src.ncattrs() else None)
            if p_rows != _rows_attr(lat_rows):
                print(f'  ERROR: {p} covers latitude rows '
                      f'{p_rows or "(full grid)"} but the merge target is '
                      f'{_rows_attr(lat_rows) or "the full grid"}; refusing '
                      f'to merge (windowed parts cannot form a full product)')
                return False
            cfg = {k: (str(src.getncattr(k)) if k in src.ncattrs()
                       else '(absent)') for k in CONFIG_ATTRS}
            timestamps.append(ts)
            plat = np.asarray(src['latitude'][:])
            plon = np.asarray(src['longitude'][:])
            pvars = [v for v in src.variables
                     if v not in ('time', 'latitude', 'longitude',
                                  GRID_MAPPING_VAR, 'crs')]
            if lat is None:
                lat, lon, var_names = plat, plon, pvars
                attrs = {k: src.getncattr(k) for k in src.ncattrs()}
                cfg0, p0 = cfg, p
            elif (not np.array_equal(lat, plat) or not np.array_equal(lon, plon)
                  or pvars != var_names):
                print(f'  ERROR: grid/variables of {p} differ from {p0}')
                return False
            elif cfg != cfg0:
                print(f'  ERROR: run configuration of {os.path.basename(p)} '
                      f'differs from {os.path.basename(p0)}; refusing to merge '
                      f'parts produced with different settings:')
                for k in CONFIG_ATTRS:
                    if cfg[k] != cfg0[k]:
                        print(f'    {k}: {cfg0[k]} vs {cfg[k]}')
                return False

    if len(month_parts) < 12:
        attrs['auseflux_partial_months'] = ','.join(
            str(m) for m, _ in month_parts)
    tmp = final_file + '.tmp'
    try:
        with _nc_full(tmp, var_names, lat, lon, timestamps,
                      attrs=attrs) as out:
            for k, p in enumerate(part_files):
                with nc4.Dataset(p) as src:
                    src.set_auto_mask(False)
                    for vn in var_names:
                        out[vn][k, :, :] = np.asarray(src[vn][0, :, :])
        os.replace(tmp, final_file)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    if verbose:
        print(f'  Merged {len(part_files)} months -> {final_file}')
    return True


def _merge_mode(selected, output_path, year_str, lat_rows, partial, verbose):
    """--merge / --merge_partial: assemble month parts into final products.
    Exits non-zero if any requested product could not be assembled."""
    failed = []
    for flux in selected:
        flux_out = os.path.join(output_path, flux)
        final_file = os.path.join(flux_out,
                                  _product_name(flux, year_str, lat_rows))
        if os.path.exists(final_file):
            print(f'{flux}: final output exists, skipping ({final_file})')
            continue
        if verbose:
            print(f'\n=== {flux}: merge month parts ===')
        parts = [(m, os.path.join(flux_out, 'parts',
                                  _part_name(flux, year_str, m, lat_rows)))
                 for m in range(1, 13)]
        have = [os.path.exists(p) for _, p in parts]
        if lat_rows is None and not (any(have) if partial else all(have)):
            win = sorted(glob.glob(os.path.join(
                flux_out, 'parts', _part_name(flux, year_str, 1)[:-3]
                + '_r*-*.nc')))
            if win:
                print(f'  ERROR: only latitude-windowed parts found (e.g. '
                      f'{os.path.basename(win[0])}); windowed parts '
                      f'cannot be merged into a full product. Use '
                      f'--merge --lat_rows START:END to build the '
                      f'windowed year file.')
                failed.append(flux)
                continue
        if partial:
            absent = [m for (m, _), h in zip(parts, have) if not h]
            parts = [mp for mp, h in zip(parts, have) if h]
            if not parts:
                print(f'  ERROR: no month part files found under '
                      f'{os.path.join(flux_out, "parts")}')
                failed.append(flux)
                continue
            if absent:
                print(f'  WARNING: --merge_partial: months '
                      f'{",".join(map(str, absent))} missing; the product '
                      f'will have {len(parts)} time steps')
        if not _merge_parts(final_file, parts, year_str, lat_rows=lat_rows,
                            verbose=verbose):
            failed.append(flux)
    if failed:
        print(f'\nERROR: merge failed for: {", ".join(failed)}')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step A: Training quantile models
# ---------------------------------------------------------------------------

def _load_training_data(training_data_path, flux, verbose=False):
    """
    Load training CSVs from OzFlux site extractions.

    Expects one CSV per site in *training_data_path*, each with columns
    matching FEATURE_COLS plus the flux target column and a 'site' column.
    """
    target_col = FLUX_METADATA[flux]['target_col']
    csv_files = sorted(f for f in os.listdir(training_data_path)
                       if f.endswith('.csv'))
    if not csv_files:
        print(f'  ERROR: no CSV files found in {training_data_path}')
        sys.exit(1)

    dfs = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(training_data_path, f),
                         index_col=0, parse_dates=True)
        # Add site identifier
        site_name = os.path.splitext(f)[0]
        df['site'] = site_name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Check target column exists
    if target_col not in data.columns:
        print(f'  ERROR: target column {target_col} not found in training data')
        print(f'  Available columns: {list(data.columns)}')
        sys.exit(1)

    # Check feature columns exist
    missing_feats = [c for c in FEATURE_COLS if c not in data.columns]
    if missing_feats:
        print(f'  ERROR: missing feature columns: {missing_feats}')
        sys.exit(1)

    # Drop rows with NaN in target or features
    subset_cols = FEATURE_COLS + [target_col]
    data = data.dropna(subset=subset_cols)

    X = data[FEATURE_COLS]
    y = data[target_col]
    sites = data['site']

    if verbose:
        print(f'  Training data: {len(X)} samples from '
              f'{sites.nunique()} sites')

    return X, y, sites


def _train_quantile_models(flux, training_data_path, models_folder,
                           n_cpus=4, verbose=False):
    """
    Train 3 quantile LightGBM models (q25, q50, q75) for one flux.

    Uses RandomizedSearchCV with TimeSeriesSplit per site, mirroring the
    original ensemble training but with quantile objective.
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import (TimeSeriesSplit, RandomizedSearchCV,
                                         KFold)
    from scipy import stats

    os.makedirs(models_folder, exist_ok=True)

    X, y, sites = _load_training_data(training_data_path, flux, verbose)

    param_grid = {
        'num_leaves': stats.randint(5, 40),
        'min_child_samples': stats.randint(10, 30),
        'max_depth': stats.randint(5, 25),
        'n_estimators': [300, 400, 500],
    }

    for q_name, alpha in QUANTILES.items():
        model_path = os.path.join(models_folder,
                                  f'{flux}_{q_name}.joblib')
        if os.path.exists(model_path):
            if verbose:
                print(f'  {flux}_{q_name}: model exists, skipping')
            continue

        if verbose:
            print(f'  Training {flux}_{q_name} (alpha={alpha})...')

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

        model = LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            boosting_type='gbdt',
            random_state=42,
            verbose=-1,
        )

        clf = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=50,
            n_jobs=n_cpus,
            cv=inner_cv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            verbose=0,
        )

        clf.fit(X, y)

        best_model = LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            boosting_type='gbdt',
            random_state=42,
            verbose=-1,
            **clf.best_params_,
        )
        best_model.fit(X, y)

        joblib.dump(best_model, model_path)

        if verbose:
            print(f'    Best params: {clf.best_params_}')
            print(f'    Saved: {model_path}')

        del clf, best_model
        gc.collect()


# ---------------------------------------------------------------------------
# Step B: Quantile prediction
# ---------------------------------------------------------------------------

def _predict_flux_quantile(flux, year, models_folder, features_path,
                           train_vars, output_path, urban_mask, lat, lon,
                           timestamps, n_lat, n_lon, mask_vars=None,
                           months=None, valid_only=True,
                           n_jobs=PREDICT_N_JOBS, lat_rows=None,
                           mask_test='nan', feat_slice='same', feat_rows=None,
                           config=None, verbose=False):
    """
    Predict 25th, 50th, 75th percentiles directly using 3 quantile models.

    Produces the same final output as step3+step4 combined, but with only
    3 model passes instead of 30.  months=None writes the full-year product;
    a list of months writes one part file per month under {output_path}/parts/.
    lat_rows=(START, END) is the FULL-grid latitude window of the outputs
    (*lat*, *urban_mask*, *n_lat* are already the window); file names get an
    _r{START}-{END} suffix and a global attribute auseflux_lat_rows.
    *feat_slice* is the row slice of the feature files ('same' = lat_rows;
    None = whole file, for features windowed with attribute *feat_rows*).

    Returns a list of messages for requested outputs that were NOT produced
    (empty = everything produced or already present).
    """
    if mask_vars is None:
        mask_vars = MASKING_VARS
    if feat_slice == 'same':
        feat_slice = lat_rows
    if config is None:
        config = _config_attrs([v for v in mask_vars if v in train_vars],
                               mask_test, valid_only)
    # 'nan': only NaN marks no-data (reference `isnull()`; +-Inf pixels are
    # predicted with Inf -> 0); 'nonfinite': NaN or +-Inf marks no-data.
    _bad = np.isnan if mask_test == 'nan' else (lambda a: ~np.isfinite(a))
    var_names = [f'{flux}_25th_percentile',
                 f'{flux}_median',
                 f'{flux}_75th_percentile']

    meta = FLUX_METADATA[flux]
    description = (
        f'AusEFlux {flux} is created by empirically upscaling '
        'the OzFlux eddy covariance network using quantile regression '
        'LightGBM models coupled with climate and remote sensing datasets. '
        'The estimates represent the 25th, 50th, and 75th percentiles from '
        'direct quantile regression.'
    )
    attrs = {
        'crs': 'EPSG:4326',
        'short_name': flux,
        'long_name': meta['long_name'],
        'units': meta['units'],
        'version': VERSION,
        'method': 'quantile_lightgbm',
        'description': description,
    }
    if lat_rows is not None:
        attrs[LAT_ROWS_ATTR] = _rows_attr(lat_rows)
    attrs.update(config)

    n_time = len(timestamps)
    failures = []

    # (output file, time indices) per write group
    if months is None:
        groups = [(os.path.join(output_path,
                                _product_name(flux, year, lat_rows)),
                   list(range(n_time)))]
    else:
        groups = []
        for m in months:
            t_idx = [t for t in range(n_time) if timestamps[t].month == m]
            if not t_idx:
                print(f'  ERROR: no time step for month {m:02d} in {year}')
                failures.append(f'no time step for month {m:02d} in {year} '
                                f'in the features')
                continue
            groups.append((os.path.join(output_path, 'parts',
                                        _part_name(flux, year, m, lat_rows)),
                           t_idx))
    groups_todo = []
    for out_file, t_idx in groups:
        if os.path.exists(out_file):
            _report_existing(out_file, config)
        else:
            groups_todo.append((out_file, t_idx))
    if not groups_todo:
        return failures

    # Load the 3 quantile models
    q_models = {}
    for q_name, alpha in QUANTILES.items():
        model_path = os.path.join(models_folder, f'{flux}_{q_name}.joblib')
        if not os.path.exists(model_path):
            print(f'  ERROR: quantile model not found: {model_path}')
            print(f'  Run with --mode train first.')
            return failures + [f'quantile model not found: {model_path}']
        q_models[q_name] = joblib.load(model_path)
        q_models[q_name].set_params(n_jobs=n_jobs)

    if verbose:
        print(f'  Loaded 3 quantile models (n_jobs={n_jobs})')

    # Open (and validate) the feature datasets lazily
    feat_datasets, err = _open_features(features_path, train_vars, year,
                                        timestamps, lat, lon,
                                        feat_slice=feat_slice,
                                        feat_rows=feat_rows)
    if err:
        print(f'  ERROR: {err}')
        return failures + [err]

    mask_var_indices = [train_vars.index(v) for v in mask_vars
                        if v in train_vars]

    n_vars = len(train_vars)
    tm = {'read': 0.0, 'predict': 0.0, 'write': 0.0}
    n_rows_pred, n_rows_total = 0, 0

    try:
        for out_file, t_idx in groups_todo:
            tmp_file = out_file + '.tmp'
            try:
                with _nc_full(tmp_file, var_names, lat, lon,
                              [timestamps[t] for t in t_idx],
                              attrs=attrs) as nc_out:
                    for k, t in enumerate(t_idx):
                        if verbose:
                            print(f'    time {t+1:02d}/{n_time}', end='\r')

                        # Allocate output arrays for all 3 quantiles
                        q_results = {q: np.full((n_lat, n_lon), np.nan,
                                                dtype='float32')
                                     for q in QUANTILES}

                        for lat_s in range(0, n_lat, TILE_ROWS):
                            lat_e = min(lat_s + TILE_ROWS, n_lat)
                            tile_h = lat_e - lat_s
                            n_pix = tile_h * n_lon

                            def _read(var):
                                return feat_datasets[var].isel(
                                    time=t, latitude=slice(lat_s, lat_e)
                                ).values.ravel().astype('float32')

                            t0 = time.perf_counter()
                            if valid_only:
                                # mask first, then build X only for valid rows
                                cache = {}
                                nodata = np.zeros(n_pix, dtype=bool)
                                for vi in mask_var_indices:
                                    cache[vi] = _read(train_vars[vi])
                                    nodata |= _bad(cache[vi])
                                urban_tile = urban_mask[lat_s:lat_e, :].ravel()
                                rows = np.flatnonzero(~nodata & ~urban_tile)
                                X = np.empty((rows.size, n_vars),
                                             dtype='float32')
                                for v, var in enumerate(train_vars):
                                    col = (cache.pop(v) if v in cache
                                           else _read(var))
                                    X[:, v] = col[rows]
                            else:
                                # Load feature tile
                                X = np.empty((n_pix, n_vars), dtype='float32')
                                for v, var in enumerate(train_vars):
                                    X[:, v] = _read(var)

                                # No-data mask
                                nodata = np.zeros(n_pix, dtype=bool)
                                for vi in mask_var_indices:
                                    nodata |= _bad(X[:, vi])

                            # Clean NaN/Inf for prediction
                            X[~np.isfinite(X)] = 0.0
                            tm['read'] += time.perf_counter() - t0
                            n_rows_pred += X.shape[0]
                            n_rows_total += n_pix

                            # Predict all 3 quantiles from the same X
                            t0 = time.perf_counter()
                            for q_name in QUANTILES:
                                if valid_only:
                                    preds = np.full(n_pix, np.nan,
                                                    dtype='float32')
                                    if rows.size:
                                        preds[rows] = q_models[q_name].predict(
                                            X).astype('float32')
                                else:
                                    preds = q_models[q_name].predict(
                                        X).astype('float32')
                                    preds[nodata] = np.nan
                                q_results[q_name][lat_s:lat_e, :] = \
                                    preds.reshape(tile_h, n_lon)
                            tm['predict'] += time.perf_counter() - t0

                            del X

                        # Apply urban/water mask and write
                        t0 = time.perf_counter()
                        for q_name in QUANTILES:
                            q_results[q_name][urban_mask] = np.nan

                        nc_out[var_names[0]][k, :, :] = q_results['q25']
                        nc_out[var_names[1]][k, :, :] = q_results['q50']
                        nc_out[var_names[2]][k, :, :] = q_results['q75']
                        tm['write'] += time.perf_counter() - t0

                        del q_results

                    if verbose:
                        print()

                os.replace(tmp_file, out_file)
            except BaseException:
                if os.path.exists(tmp_file):  # never leave a partial .tmp
                    os.remove(tmp_file)
                raise
            if verbose:
                print(f'  Output: {out_file}')
    finally:
        # Close feature datasets
        for ds_da in feat_datasets.values():
            ds_da.close()

    # Unload models
    del q_models
    gc.collect()

    if verbose:
        frac = n_rows_pred / n_rows_total if n_rows_total else 0.0
        print(f'  timing {flux}: read {tm["read"]:.1f}s | predict '
              f'{tm["predict"]:.1f}s | write {tm["write"]:.1f}s | rows '
              f'predicted {frac:.1%} | peak RSS {_peak_rss_gb():.2f} GB')
    return failures


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_quantile_pipeline(year, mode='predict', training_data_path=None,
                          features_path=None, static_path=None,
                          results_path=None, fluxes=None, n_cpus=4,
                          verbose=False, models_path=None,
                          variables_file=None, output_path=None,
                          mask_vars=None, months=None, merge=False,
                          valid_only=True, threads=PREDICT_N_JOBS,
                          lat_rows=None, mask_test='nan', merge_partial=False):
    """
    lat_rows=(START, END): predict only FULL-grid rows START..END-1;
    latitude-windowed features (auseflux_lat_rows attribute) are detected
    automatically (outputs and urban mask follow their window; lat_rows must
    lie inside it).  Exits non-zero if any requested output was not produced
    (existing outputs count as done).
    models_path / output_path are base directories; the flux name is appended
    (defaults {results}/models/quantile/v2.1 and {results}/AusEFlux_quantile).
    """
    if features_path is None:
        features_path = DEFAULT_FEATURES
    if static_path is None:
        static_path = DEFAULT_STATIC
    if results_path is None:
        results_path = DEFAULT_RESULTS
    if models_path is None:
        models_path = os.path.join(results_path, 'models', 'quantile', VERSION)
    if output_path is None:
        output_path = os.path.join(results_path, 'AusEFlux_quantile')
    if variables_file is None:
        variables_file = os.path.join(results_path, 'variables.txt')
    if mask_vars is None:
        mask_vars = MASKING_VARS

    year_str = str(year)
    selected = fluxes if fluxes else list(FLUX_METADATA.keys())

    # --- Training mode ---
    if mode == 'train':
        if training_data_path is None:
            print('ERROR: --training_data is required for training mode')
            sys.exit(1)

        if not os.path.isdir(training_data_path):
            print(f'ERROR: training data path not found: {training_data_path}')
            sys.exit(1)

        for flux in selected:
            if flux not in FLUX_METADATA:
                print(f'WARNING: unknown flux {flux}, skipping')
                continue
            if verbose:
                print(f'\n=== Training quantile models for {flux} ===')
            _train_quantile_models(flux, training_data_path,
                                   os.path.join(models_path, flux),
                                   n_cpus=n_cpus, verbose=verbose)

        if verbose:
            print('\nTraining complete.')
        return

    # --- Merge mode: assemble month parts into final products ---
    if merge or merge_partial:
        _merge_mode(selected, output_path, year_str, lat_rows,
                    partial=merge_partial, verbose=verbose)
        return

    # --- Prediction mode ---
    features_list = variables_file
    if not os.path.exists(features_list):
        print(f'ERROR: variables.txt not found: {features_list}')
        sys.exit(1)

    train_vars = _parse_variables(features_list)
    if verbose:
        print(f'Feature variables ({len(train_vars)}): '
              f'{", ".join(train_vars)}')

    eff_mask = [v for v in mask_vars if v in train_vars]
    ignored = [v for v in mask_vars if v not in train_vars]
    print(f'Mask variables: {", ".join(eff_mask) or "(none)"}'
          + (f'  (ignored, not in variables file: {", ".join(ignored)})'
             if ignored else ''))
    if verbose:
        print(f'Pixels: {"valid_only" if valid_only else "all_pixels"} | '
              f'months: {",".join(map(str, months)) if months else "all"} | '
              f'threads: {threads} | mask_test: {mask_test} | lat_rows: '
              f'{_rows_attr(lat_rows) or "all"}')

    missing = [v for v in train_vars
               if not os.path.exists(
                   os.path.join(features_path, f'{v}_{TARGET_GRID}.nc'))]
    if missing:
        print(f'ERROR: missing feature files: {", ".join(missing)}')
        sys.exit(1)

    # Grid / time axis / latitude window from the first feature file, then
    # validate every feature against it (time axis, window, coordinates).
    g = _feature_grid(features_path, train_vars[0], year_str, lat_rows,
                      verbose=verbose)
    lat, lon, timestamps = g['lat'], g['lon'], g['timestamps']
    out_rows = g['out_rows']
    n_lat, n_lon = len(lat), len(lon)
    feats, err = _open_features(features_path, train_vars, year_str,
                                timestamps, lat, lon,
                                feat_slice=g['feat_slice'],
                                feat_rows=g['feat_rows'])
    if err:
        print(f'ERROR: {err}')
        sys.exit(1)
    for da in feats.values():
        da.close()

    # Urban/water mask for the same full-grid rows (coordinates checked)
    urban_mask = _load_urban_mask(static_path, out_rows, lat, lon)
    config = _config_attrs(eff_mask, mask_test, valid_only)

    # Process each flux
    failures = []
    for flux in selected:
        if flux not in FLUX_METADATA:
            print(f'ERROR: unknown flux {flux}')
            failures.append((flux, 'unknown flux'))
            continue

        models_folder = os.path.join(models_path, flux)
        flux_out = os.path.join(output_path, flux)

        if not os.path.isdir(models_folder):
            print(f'ERROR: quantile models folder not found: '
                  f'{models_folder}')
            print(f'  Run with --mode train first.')
            failures.append((flux, f'quantile models folder not found: '
                                   f'{models_folder}'))
            continue

        if verbose:
            print(f'\n=== Predict {flux} (quantile) ===')

        # A finished full-year product makes month parts unnecessary
        final_file = os.path.join(flux_out,
                                  _product_name(flux, year_str, out_rows))
        if months is not None and os.path.exists(final_file):
            _report_existing(final_file, config)
            continue

        fl = _predict_flux_quantile(
            flux=flux,
            year=year_str,
            models_folder=models_folder,
            features_path=features_path,
            train_vars=train_vars,
            output_path=flux_out,
            urban_mask=urban_mask,
            lat=lat, lon=lon,
            timestamps=timestamps,
            n_lat=n_lat, n_lon=n_lon,
            mask_vars=mask_vars,
            months=months,
            valid_only=valid_only,
            n_jobs=threads,
            lat_rows=out_rows,
            mask_test=mask_test,
            feat_slice=g['feat_slice'],
            feat_rows=g['feat_rows'],
            config=config,
            verbose=verbose,
        )
        failures += [(flux, f) for f in fl]

        gc.collect()

    if failures:
        print('\nERROR: requested outputs were NOT produced:')
        for flux, msg in failures:
            print(f'  {flux}: {msg}')
        sys.exit(1)
    if verbose:
        print('\nAll fluxes complete.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux: Quantile LightGBM prediction pipeline.')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (e.g. 2024).')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'],
                        default='predict',
                        help='Mode: train quantile models or predict '
                             '(default: predict).')
    parser.add_argument('--training_data', type=str, default=None,
                        help='Path to training CSVs (required for '
                             '--mode train).')
    parser.add_argument('--features_path', type=str, default=None,
                        help=f'Feature datasets directory. '
                             f'Default: {DEFAULT_FEATURES} '
                             f'($AUSEFLUX_ROOT/features_500m)')
    parser.add_argument('--static_path', type=str, default=None,
                        help=f'Static input files. Default: {DEFAULT_STATIC} '
                             f'($AUSEFLUX_STATIC)')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Results directory. Default: {DEFAULT_RESULTS}')
    parser.add_argument('--models_path', type=str, default=None,
                        help='Base folder of quantile models '
                             '({models_path}/{flux}/{flux}_q25.joblib ...). '
                             'Default: {results}/models/quantile/v2.1')
    parser.add_argument('--variables_file', type=str, default=None,
                        help='Feature list (variables.txt). Default: '
                             '{results}/variables.txt')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Base output folder; products are written to '
                             '{output_path}/{flux}/. Default: '
                             '{results}/AusEFlux_quantile')
    parser.add_argument('--fluxes', type=str, nargs='+',
                        choices=['GPP', 'NEE', 'ER', 'ET'], default=None,
                        help='Fluxes to process (default: all).')
    parser.add_argument('--n_cpus', type=int, default=4,
                        help='CPUs for training hyperparameter search '
                             '(default: 4).')
    parser.add_argument('--mask_vars', type=str,
                        default=','.join(MASKING_VARS),
                        help='Comma list of features whose NaN marks no-data '
                             f'(default: {",".join(MASKING_VARS)} = reference; '
                             'previous default: VegH,NDWI,rain_anom,Tavg).')
    parser.add_argument('--mask_test', choices=['nan', 'nonfinite'],
                        default='nan',
                        help='What marks a mask-variable pixel as no-data: '
                             'nan (default) = NaN only, like the reference '
                             '(`isnull()`), so +-Inf pixels are predicted with '
                             'Inf -> 0; nonfinite = NaN or +-Inf (previous '
                             'behaviour).')
    parser.add_argument('--months', type=str, default=None,
                        help='Month subset to predict, e.g. "1-6" or '
                             '"1,6,12". When given, writes one part file per '
                             'month to {output}/{flux}/parts/..._m{MM}.nc '
                             '(combine with --merge). Default: all months.')
    parser.add_argument('--merge', action='store_true',
                        help='Assemble the 12 month part files into the final '
                             'product (parts are kept) and exit. Parts must '
                             'share the run-configuration attributes.')
    parser.add_argument('--merge_partial', action='store_true',
                        help='Like --merge, but accept fewer than 12 month '
                             'parts: the product time dimension is the number '
                             'of parts (listed in the auseflux_partial_months '
                             'attribute) and a WARNING names the missing '
                             'months.')
    pix = parser.add_mutually_exclusive_group()
    pix.add_argument('--valid_only', dest='valid_only', action='store_true',
                     help='Predict only pixels that are not no-data/urban and '
                          'scatter back (default; bit-identical output).')
    pix.add_argument('--all_pixels', dest='valid_only', action='store_false',
                     help='Predict every pixel then mask (previous default; '
                          'same output, slower).')
    parser.set_defaults(valid_only=True)
    parser.add_argument('--threads', type=int, default=PREDICT_N_JOBS,
                        help='n_jobs for LightGBM predict (default: -1 = all '
                             'cores; 0 is invalid).')
    parser.add_argument('--lat_rows', type=str, default=None,
                        metavar='START:END',
                        help='Predict only latitude rows START..END-1 of the '
                             'FULL grid (0-based, END exclusive, e.g. '
                             '800:1200). Outputs (product, month parts) get an '
                             '_r{START}-{END} suffix and a global attribute '
                             'auseflux_lat_rows. If the features are windowed '
                             '(auseflux_lat_rows attribute), START:END must lie '
                             'inside their window; without --lat_rows their '
                             'window is used. With --merge, merges the '
                             'windowed parts of that window. Default: full '
                             'grid.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Progress and timing output (always on).')
    args = parser.parse_args()

    if (args.merge or args.merge_partial) and args.months:
        parser.error('--merge/--merge_partial cannot be combined with --months')
    if args.threads == 0:
        parser.error('--threads must be -1 (all cores) or >= 1')
    try:
        months = _parse_months(args.months) if args.months else None
        lat_rows = _parse_lat_rows(args.lat_rows) if args.lat_rows else None
    except ValueError as e:
        parser.error(str(e))
    mask_vars = [v.strip() for v in args.mask_vars.split(',') if v.strip()]

    run_quantile_pipeline(
        year=args.year,
        mode=args.mode,
        training_data_path=args.training_data,
        features_path=args.features_path,
        static_path=args.static_path,
        results_path=args.results_path,
        fluxes=args.fluxes,
        n_cpus=args.n_cpus,
        verbose=args.verbose,
        models_path=args.models_path,
        variables_file=args.variables_file,
        output_path=args.output_path,
        mask_vars=mask_vars,
        months=months,
        merge=args.merge,
        merge_partial=args.merge_partial,
        valid_only=args.valid_only,
        threads=args.threads,
        lat_rows=lat_rows,
        mask_test=args.mask_test,
    )


if __name__ == '__main__':
    main()
