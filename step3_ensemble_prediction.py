"""
Step 3 (+4): Fused ensemble prediction and combination for AusEFlux.

For each flux, every pre-trained ensemble member (LightGBM + RandomForest) is
applied to the feature datasets for the target year and the 25th/50th/75th
percentile bands are computed directly, writing a single output product.

This is a *fused* predict+combine: features for each (time, latitude-band) tile
are read ONCE and all ensemble members predict on them, then the per-tile
quantiles are written straight to the final product. The earlier design read
every feature once per model (30x), wrote one file per model, and re-read those
files to combine; the fused path removes that redundant I/O and produces
bit-identical output (the no-data / urban masks are identical across members, so
the per-tile quantile over the members equals the previous write-then-quantile).

Optionally, LightGBM members can be pre-compiled with treelite/tl2cgen for a
~1.5x exact predict speedup (see benchmark/compile_models.py): if a
`<models_folder>/compiled/<member>.so` exists it is used instead of the joblib.
RandomForest members are NOT compiled (treelite's codegen explodes on their deep
trees and GTIL is slower) and always load from joblib.

Memory: each tile holds one feature matrix plus the per-member prediction stack,
bounded by TILE_ROWS.

Paths default to $AUSEFLUX_ROOT (default /home/prl900/Downloads/wald_eflux):
features {root}/features_500m, static $AUSEFLUX_STATIC ({root}/input/static),
results {root}/results (variables.txt, models/ensemble/v2.1/{flux},
AusEFlux/{flux}); each is overridable on the command line.

Defaults reproduce the published v2.1 product exactly (Gadi experiments E1,
E2a: bit-identical to the published 2024/2025 products) and are also the
fastest configuration:
  --quantile_method nearest  nearest-rank percentile = the reference
        `np_percentile` (master:src/_percentile.py): sort the members, take
        index round_half_even(q*(n_valid-1)), NaN where < 3 valid members.
        `linear` (np.nanquantile) is kept for comparison only: it biases the
        bands (median +0.2..0.3 per month, NEE annual sink -3 %) and is ~200x
        slower.
  --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg  the reference no-data
        variables (dropping LAI_anom predicts ~0.4 % more pixels).
  --mask_test nan  only NaN marks no-data, like the reference `isnull()`
        (+-Inf, e.g. LAI_anom where the climatology is 0, is predicted with
        Inf -> 0); `nonfinite` also masks +-Inf.
  --valid_only  build X only for unmasked, non-urban pixels and scatter back
        (bit-identical to --all_pixels, ~1.8x faster predict).
Other switches: --n_models, --threads, --dump_members DIR.
  --months 1-6 writes per-month part files under {output}/parts/; a later
        --merge run assembles the 12 parts into the final product
        (--merge_partial: whatever months exist, with a warning).
  --lat_rows START:END processes a latitude window only (rows START..END-1
        of the FULL grid); outputs get an _r{START}-{END} suffix (also parts
        and member dumps).  Latitude-windowed features (step2 --lat_rows,
        global attribute auseflux_lat_rows) are detected automatically: the
        outputs are named after, and the urban mask sliced to, their window.

Safety: every feature must have the same time axis, latitude window and
coordinates as the first one, and the urban/water mask must match the
feature grid (1e-4 deg), else the run stops with an error.  Products, parts
and member dumps carry the run configuration as global attributes
(quantile_method, mask_vars, mask_test, n_models, valid_only, lat_rows);
parts with different configurations are not merged, and an existing output
with a different configuration is skipped with a WARNING.  The exit status
is non-zero if any requested output was not produced (existing outputs
count as done).

Usage:
    uv run step3_ensemble_prediction.py --year 2024
    uv run step3_ensemble_prediction.py --year 2024 --fluxes GPP ET
    uv run step3_ensemble_prediction.py --year 2024 --fluxes GPP --months 1-6
    uv run step3_ensemble_prediction.py --year 2024 --fluxes GPP --merge
    uv run step3_ensemble_prediction.py --selftest
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
# To use treelite-compiled LightGBM members, also install "tl2cgen".

import argparse
import gc
import glob
import os
import re
import resource
import sys
import time
import warnings
from contextlib import ExitStack, contextmanager

warnings.filterwarnings('ignore')

import joblib
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants / path resolution
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

QUANTILES = (0.25, 0.50, 0.75)

FLUX_METADATA = {
    'GPP': {
        'long_name': 'Gross Primary Productivity',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
    },
    'NEE': {
        'long_name': 'Net Ecosystem Exchange',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
    },
    'ER': {
        'long_name': 'Ecosystem Respiration',
        'units': 'gC/m\N{SUPERSCRIPT TWO}/month',
    },
    'ET': {
        'long_name': 'Evapotranspiration',
        'units': 'mm/month',
    },
}

TILE_ROWS = 1400  # latitude rows per tile (multiple of the 200-row output chunks)
PREDICT_N_JOBS = -1  # LightGBM/RandomForest predict threads (-1 = all cores)


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
    """
    Parse variables.txt into an ordered list of feature variable names.

    The file is a single CSV header line like:
        LST_RS,ΔT_RS,...,VegH_EC,site,
    We strip the _RS/_EC suffix and drop 'site'.
    """
    with open(features_list) as f:
        header = f.read().strip().split(',')
    # Remove empty strings and 'site'
    train_vars = [v for v in header if v and v != 'site']
    # Strip 3-character suffix (_RS or _EC)
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

# Run-configuration global attributes written on products, month parts and
# member dumps.  Month parts must agree on all of them to be merged, and an
# existing output whose values differ from the request is reported (WARNING)
# when it is skipped.
CONFIG_ATTRS = ('quantile_method', 'mask_vars', 'mask_test', 'n_models',
                'valid_only', 'lat_rows')


def _config_attrs(quantile_method, mask_vars, mask_test, n_models,
                  valid_only, lat_rows):
    """Run-configuration attributes (see CONFIG_ATTRS).  *mask_vars* are the
    effective mask variables, *n_models* the number of members used and
    *lat_rows* the full-grid latitude window ('all' = full grid)."""
    return {
        'quantile_method': quantile_method,
        'mask_vars': ','.join(mask_vars),
        'mask_test': mask_test,
        'n_models': int(n_models),
        'valid_only': int(bool(valid_only)),
        'lat_rows': _rows_attr(lat_rows) or 'all',
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
# Quantile combination
# ---------------------------------------------------------------------------

def _nearest_rank_quantiles(stack, qs):
    """
    Nearest-rank percentiles over axis 0, reproducing the reference
    `np_percentile(xx, q, nodata=np.nan)` (master:src/_percentile.py) exactly.

    *stack* is (n_models, n_pix) with NaN for invalid members.  Per pixel:
    sort along the member axis (NaNs sort to the end), take index
    round_half_even(q * (n_valid - 1)), and set NaN where n_valid < 3.
    Returns (len(qs), n_pix) in the dtype of *stack*.
    """
    stack = np.asarray(stack)
    n_pix = stack.shape[1]
    valid_counts = (~np.isnan(stack)).sum(axis=0)
    out = np.full((len(qs), n_pix), np.nan, dtype=stack.dtype)
    cols = np.flatnonzero(valid_counts >= 3)
    if cols.size:
        srt = np.sort(stack[:, cols], axis=0)
        vc = valid_counts[cols]
        ar = np.arange(cols.size)
        for k, q in enumerate(qs):
            idx = np.round(q * (vc - 1)).astype(np.int64)
            out[k, cols] = srt[idx, ar]
    return out


def _combine(stack, method):
    """25/50/75 percentiles over members (axis 0); NaN where < 3 valid."""
    if method == 'nearest':
        return _nearest_rank_quantiles(stack, QUANTILES)
    valid_count = np.sum(np.isfinite(stack), axis=0)
    q = np.nanquantile(stack, list(QUANTILES), axis=0).astype('float32')
    q[:, valid_count < 3] = np.nan
    return q


# ---------------------------------------------------------------------------
# NetCDF4 output writer (full dimensions known upfront)
# ---------------------------------------------------------------------------
# CRS variable of the published v2.1 products (odc-geo `assign_crs` on
# EPSG:4326): an int scalar `spatial_ref` with the WKT2 string, referenced by
# `grid_mapping` on every data variable.
SPATIAL_REF_ATTRS = {
    'spatial_ref': 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
    'crs_wkt': 'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",MEMBER["World Geodetic System 1984 (Transit)"],MEMBER["World Geodetic System 1984 (G730)"],MEMBER["World Geodetic System 1984 (G873)"],MEMBER["World Geodetic System 1984 (G1150)"],MEMBER["World Geodetic System 1984 (G1674)"],MEMBER["World Geodetic System 1984 (G1762)"],MEMBER["World Geodetic System 1984 (G2139)"],MEMBER["World Geodetic System 1984 (G2296)"],ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],USAGE[SCOPE["Horizontal component of 3D system."],AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]',
    'semi_major_axis': 6378137.0,
    'semi_minor_axis': 6356752.31424518,
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
    """
    Create output NetCDF with multiple variables sharing the same grid.
    *var_names* is a list of variable names to create.

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
                    continue  # netCDF4 doesn't like NaN attrs
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
# Ensemble member predictors (joblib, with optional treelite-compiled LightGBM)
# ---------------------------------------------------------------------------

def _clean_pred(raw):
    """Non-finite predictions -> 0 (reference predict_xr clean=True), float32."""
    raw = np.asarray(raw).ravel()
    fin = np.isfinite(raw)
    if not fin.all():
        raw = np.where(fin, raw, 0)
    return raw.astype('float32')


def _member_kind(name, model=None):
    """'lgbm' / 'rf' label used for timing breakdowns."""
    tn = type(model).__name__.lower() if model is not None else ''
    if 'lgbm' in tn or 'lightgbm' in tn or 'booster' in tn:
        return 'lgbm'
    if 'forest' in tn:
        return 'rf'
    low = name.lower()
    if 'lgbm' in low:
        return 'lgbm'
    if '_rf' in low or low.startswith('rf'):
        return 'rf'
    return tn or 'other'


def _member_files(models_folder, n_models=None):
    """Ensemble member joblib files, sorted by name; the first *n_models*."""
    files = sorted(f for f in os.listdir(models_folder)
                   if f.endswith('.joblib'))
    return files[:n_models] if n_models else files


def _usable_cpus():
    """CPUs this process may run on (PBS/cgroup affinity), not the node's."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def _load_predictors(models_folder, n_models=None, n_jobs=PREDICT_N_JOBS,
                     verbose=False):
    """
    Return a list of (name, kind, predict) tuples, one per ensemble member,
    where predict(X) -> 1d float32 and name is the joblib stem
    (e.g. 'GPP_lgbm_1').  Members are sorted by file name; *n_models* keeps
    the first N.

    For each member, a compiled `compiled/<member>.so` (treelite/tl2cgen) is used
    if present (LightGBM only, exact, ~1.5x faster); otherwise the joblib model is
    loaded.  RandomForest members are always joblib (treelite does not help them).
    """
    model_files = _member_files(models_folder, n_models)
    compiled_dir = os.path.join(models_folder, 'compiled')
    nthread = _usable_cpus() if (n_jobs is None or n_jobs < 1) else n_jobs
    members, n_compiled = [], 0
    for mf in model_files:
        name = mf[:-len('.joblib')]
        so = os.path.join(compiled_dir, name + '.so')
        if os.path.exists(so):
            import tl2cgen  # lazy: only needed when compiled members exist
            pr = tl2cgen.Predictor(so, nthread=nthread)
            members.append((name, 'lgbm',
                            lambda X, pr=pr: _clean_pred(
                                pr.predict(tl2cgen.DMatrix(X)))))
            n_compiled += 1
        else:
            m = joblib.load(os.path.join(models_folder, mf))
            try:
                m.set_params(n_jobs=n_jobs)
            except Exception:
                pass
            members.append((name, _member_kind(name, m),
                            lambda X, m=m: _clean_pred(m.predict(X))))
    if verbose:
        extra = f' ({n_compiled} treelite-compiled)' if n_compiled else ''
        kinds = pd.Series([k for _, k, _ in members]).value_counts().to_dict() \
            if members else {}
        print(f'  {len(members)} ensemble members{extra} {kinds}, '
              f'n_jobs={n_jobs}')
    return members


# ---------------------------------------------------------------------------
# Fused predict + combine
# ---------------------------------------------------------------------------

def _predict_and_combine(flux, year, models_folder, features_path, train_vars,
                         output_path, urban_mask, lat, lon, timestamps,
                         n_lat, n_lon, mask_vars=None, quantile_method='nearest',
                         months=None, valid_only=True, dump_members=None,
                         n_models=None, n_jobs=PREDICT_N_JOBS,
                         mask_test='nan', lat_rows=None, feat_slice='same',
                         feat_rows=None, config=None, verbose=False):
    """
    Predict all ensemble members and write the 25/50/75-percentile product in a
    single pass: features per (time, latitude-band) tile are read once, every
    member predicts on them, and the per-tile quantiles are written directly.

    months=None writes the full-year product; a list of months writes one part
    file per month under {output_path}/parts/ (and member dumps likewise).

    lat_rows=(START, END) is the FULL-grid latitude window of the outputs
    (*lat*, *urban_mask* and *n_lat* must already be the window; see
    run_ensemble_pipeline): file names get an _r{START}-{END} suffix and a
    global attribute auseflux_lat_rows='START:END'.  *feat_slice* is the
    corresponding row slice of the feature files ('same' = lat_rows, i.e.
    full-grid features; None = whole file, for features already windowed
    with attribute *feat_rows*).  *config* = run-configuration attributes
    (CONFIG_ATTRS) written on products, parts and dumps.

    Returns a list of messages for requested outputs that were NOT produced
    (empty = everything produced or already present).
    """
    if mask_vars is None:
        mask_vars = MASKING_VARS
    if feat_slice == 'same':
        feat_slice = lat_rows
    # 'nan': only NaN marks no-data (reference `isnull()`; +-Inf pixels are
    # predicted with Inf -> 0); 'nonfinite': NaN or +-Inf marks no-data.
    _bad = np.isnan if mask_test == 'nan' else (lambda a: ~np.isfinite(a))

    # Open (and validate) the feature datasets lazily for the target year.
    feat_datasets, err = _open_features(features_path, train_vars, year,
                                        timestamps, lat, lon,
                                        feat_slice=feat_slice,
                                        feat_rows=feat_rows)
    if err:
        print(f'  ERROR: {err}')
        return [err]
    try:
        return _predict_groups(
            flux, year, models_folder, train_vars, output_path, urban_mask,
            lat, lon, timestamps, n_lat, n_lon, mask_vars, quantile_method,
            months, valid_only, dump_members, n_models, n_jobs, _bad,
            lat_rows, config, feat_datasets, verbose)
    finally:
        for da in feat_datasets.values():
            da.close()


def _predict_groups(flux, year, models_folder, train_vars, output_path,
                    urban_mask, lat, lon, timestamps, n_lat, n_lon, mask_vars,
                    quantile_method, months, valid_only, dump_members,
                    n_models, n_jobs, _bad, lat_rows, config, feat_datasets,
                    verbose):
    """Body of _predict_and_combine (features already open and validated)."""
    members = _load_predictors(models_folder, n_models=n_models,
                               n_jobs=n_jobs, verbose=verbose)
    n_members = len(members)
    if n_members == 0:
        print(f'  ERROR: no models found in {models_folder}')
        return [f'no models found in {models_folder}']

    mask_var_indices = [train_vars.index(v) for v in mask_vars
                        if v in train_vars]
    n_time = len(timestamps)
    n_vars = len(train_vars)
    if config is None:
        config = _config_attrs(quantile_method,
                               [v for v in mask_vars if v in train_vars],
                               'nan' if _bad is np.isnan else 'nonfinite',
                               n_members, valid_only, lat_rows)

    var_names = [f'{flux}_25th_percentile',
                 f'{flux}_median',
                 f'{flux}_75th_percentile']
    meta = FLUX_METADATA[flux]
    attrs = {
        'crs': 'EPSG:4326',
        'short_name': flux,
        'long_name': meta['long_name'],
        'units': meta['units'],
        'version': VERSION,
        'description': (
            f'AusEFlux {flux} is created by empirically upscaling '
            'the OzFlux eddy covariance network using machine learning methods '
            'coupled with climate and remote sensing datasets. The estimates '
            'provided within this dataset were extracted from an ensemble of '
            'predictions and represent the median and uncertainty range.'),
    }
    if lat_rows is not None:
        attrs[LAT_ROWS_ATTR] = _rows_attr(lat_rows)
    attrs.update(config)
    sfx = _rows_suffix(lat_rows)

    # (output file, time indices, member dump files) per write group
    failures = []
    dump_dir = os.path.join(dump_members, flux) if dump_members else None
    groups = []
    if months is None:
        dumps = ([os.path.join(dump_dir, f'{name}{sfx}.nc')
                  for name, _, _ in members] if dump_dir else [])
        groups.append((os.path.join(output_path,
                                    _product_name(flux, year, lat_rows)),
                       list(range(n_time)), dumps))
    else:
        for m in months:
            t_idx = [t for t in range(n_time) if timestamps[t].month == m]
            if not t_idx:
                print(f'  ERROR: no time step for month {m:02d} in {year}')
                failures.append(f'no time step for month {m:02d} in {year} '
                                f'in the features')
                continue
            dumps = ([os.path.join(dump_dir, 'parts',
                                   f'{name}_m{m:02d}{sfx}.nc')
                      for name, _, _ in members] if dump_dir else [])
            groups.append((os.path.join(output_path, 'parts',
                                        _part_name(flux, year, m, lat_rows)),
                           t_idx, dumps))

    tm = {'read': 0.0, 'quantile': 0.0, 'write': 0.0}
    tm_pred = {}
    n_rows_pred, n_rows_total = 0, 0

    for out_file, t_idx, dump_files in groups:
        if os.path.exists(out_file):
            _report_existing(out_file, config)
            gone = [p for p in dump_files if not os.path.exists(p)]
            if gone:
                msg = (f'{out_file} exists but {len(gone)} member dump(s) '
                       f'requested with --dump_members are missing (e.g. '
                       f'{gone[0]}); dumps are only written together with '
                       f'the product: remove it or use another --output_path')
                print(f'  ERROR: {msg}')
                failures.append(msg)
            continue
        group_times = [timestamps[t] for t in t_idx]
        tmp_files = [out_file + '.tmp'] + [p + '.tmp' for p in dump_files]
        try:
            with ExitStack() as es:
                nc_out = es.enter_context(
                    _nc_full(tmp_files[0], var_names, lat, lon, group_times,
                             attrs=attrs))
                nc_dumps = [
                    es.enter_context(_nc_full(
                        tmp, [flux], lat, lon, group_times,
                        attrs={'crs': 'EPSG:4326', 'short_name': flux,
                               'long_name': meta['long_name'],
                               'units': meta['units'], 'version': VERSION,
                               'member': name,
                               **({LAT_ROWS_ATTR: _rows_attr(lat_rows)}
                                  if lat_rows is not None else {}),
                               **config}))
                    for tmp, (name, _, _) in zip(tmp_files[1:], members)]

                for k, t in enumerate(t_idx):
                    if verbose:
                        print(f'    time {t+1:02d}/{n_time}', end='\r')

                    for lat_s in range(0, n_lat, TILE_ROWS):
                        lat_e = min(lat_s + TILE_ROWS, n_lat)
                        tile_h = lat_e - lat_s
                        n_pix = tile_h * n_lon
                        urban_tile = urban_mask[lat_s:lat_e, :].ravel()

                        def _read(var):
                            return feat_datasets[var].isel(
                                time=t, latitude=slice(lat_s, lat_e)
                            ).values.ravel().astype('float32')

                        # --- read features ONCE for this tile ---
                        t0 = time.perf_counter()
                        if valid_only:
                            # mask first, then build X only for valid rows
                            cache = {}
                            nodata = np.zeros(n_pix, dtype=bool)
                            for vi in mask_var_indices:
                                cache[vi] = _read(train_vars[vi])
                                nodata |= _bad(cache[vi])
                            rows = np.flatnonzero(~nodata & ~urban_tile)
                            X = np.empty((rows.size, n_vars), dtype='float32')
                            for v, var in enumerate(train_vars):
                                col = cache.pop(v) if v in cache else _read(var)
                                X[:, v] = col[rows]
                        else:
                            X = np.empty((n_pix, n_vars), dtype='float32')
                            for v, var in enumerate(train_vars):
                                X[:, v] = _read(var)
                            # No-data mask: any masking var is NaN
                            nodata = np.zeros(n_pix, dtype=bool)
                            for vi in mask_var_indices:
                                nodata |= _bad(X[:, vi])
                        X[~np.isfinite(X)] = 0.0  # NaN/Inf -> 0 for prediction
                        tm['read'] += time.perf_counter() - t0
                        n_rows_pred += X.shape[0]
                        n_rows_total += n_pix

                        # --- predict every member on the same X ---
                        if valid_only:
                            stack = np.full((n_members, n_pix), np.nan,
                                            dtype='float32')
                        else:
                            stack = np.empty((n_members, n_pix),
                                             dtype='float32')
                        for mi, (_, kind, predict) in enumerate(members):
                            t0 = time.perf_counter()
                            if valid_only:
                                if rows.size:
                                    stack[mi, rows] = predict(X)
                            else:
                                stack[mi] = predict(X)
                            tm_pred[kind] = (tm_pred.get(kind, 0.0)
                                             + time.perf_counter() - t0)
                        if not valid_only:
                            stack[:, nodata] = np.nan
                            stack[:, urban_tile] = np.nan

                        # --- combine: 25/50/75 percentiles (>= 3 valid) ---
                        t0 = time.perf_counter()
                        q = _combine(stack, quantile_method)
                        tm['quantile'] += time.perf_counter() - t0

                        t0 = time.perf_counter()
                        for b in range(3):
                            nc_out[var_names[b]][k, lat_s:lat_e, :] = \
                                q[b].reshape(tile_h, n_lon)
                        for mi, nd in enumerate(nc_dumps):
                            nd[flux][k, lat_s:lat_e, :] = \
                                stack[mi].reshape(tile_h, n_lon)
                        tm['write'] += time.perf_counter() - t0
                        del X, stack, q
                        gc.collect()

            for tmp in tmp_files:
                os.replace(tmp, tmp[:-len('.tmp')])
        except BaseException:
            for tmp in tmp_files:  # never leave partial .tmp files behind
                if os.path.exists(tmp):
                    os.remove(tmp)
            raise
        if verbose:
            print()
            print(f'  Output: {out_file}')

    if verbose:
        pk = ', '.join(f'{k} {v:.1f}s' for k, v in sorted(tm_pred.items()))
        frac = n_rows_pred / n_rows_total if n_rows_total else 0.0
        print(f'  timing {flux}: read {tm["read"]:.1f}s | predict [{pk}] | '
              f'quantile({quantile_method}) {tm["quantile"]:.1f}s | '
              f'write {tm["write"]:.1f}s | rows predicted {frac:.1%} | '
              f'peak RSS {_peak_rss_gb():.2f} GB')
    return failures


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_ensemble_pipeline(year, features_path=None, static_path=None,
                          results_path=None, fluxes=None, verbose=False,
                          models_path=None, variables_file=None,
                          output_path=None, mask_vars=None,
                          quantile_method='nearest', months=None, merge=False,
                          valid_only=True, dump_members=None, n_models=None,
                          threads=PREDICT_N_JOBS, mask_test='nan',
                          lat_rows=None, merge_partial=False):
    """
    Run the fused ensemble predict+combine for all requested fluxes.

    lat_rows=(START, END): process only FULL-grid rows START..END-1 (outputs
    carry an _r{START}-{END} suffix; see _predict_and_combine).  Features
    that are themselves latitude-windowed (step2 --lat_rows, global
    attribute auseflux_lat_rows) are handled automatically: outputs and the
    urban mask follow their window, and lat_rows must lie inside it.

    models_path / output_path are base directories; the flux name is appended
    (defaults {results}/models/ensemble/v2.1 and {results}/AusEFlux).

    Exits non-zero (after processing the other fluxes) if any requested
    output was not produced; outputs that already exist count as done.
    """
    if features_path is None:
        features_path = DEFAULT_FEATURES
    if static_path is None:
        static_path = DEFAULT_STATIC
    if results_path is None:
        results_path = DEFAULT_RESULTS
    if models_path is None:
        models_path = os.path.join(results_path, 'models', 'ensemble', VERSION)
    if output_path is None:
        output_path = os.path.join(results_path, 'AusEFlux')
    if variables_file is None:
        variables_file = os.path.join(results_path, 'variables.txt')
    if mask_vars is None:
        mask_vars = MASKING_VARS

    year_str = str(year)
    selected = fluxes if fluxes else list(FLUX_METADATA.keys())

    # --- Merge mode: assemble month parts into final products ---
    if merge or merge_partial:
        _merge_mode(selected, output_path, year_str, lat_rows,
                    partial=merge_partial, verbose=verbose)
        return

    # --- Check inputs ---
    features_list = variables_file
    if not os.path.exists(features_list):
        print(f'ERROR: variables.txt not found: {features_list}')
        sys.exit(1)

    train_vars = _parse_variables(features_list)
    if verbose:
        print(f'Feature variables ({len(train_vars)}): {", ".join(train_vars)}')

    eff_mask = [v for v in mask_vars if v in train_vars]
    ignored = [v for v in mask_vars if v not in train_vars]
    print(f'Mask variables: {", ".join(eff_mask) or "(none)"}'
          + (f'  (ignored, not in variables file: {", ".join(ignored)})'
             if ignored else ''))
    if verbose:
        print(f'Quantile method: {quantile_method} | pixels: '
              f'{"valid_only" if valid_only else "all_pixels"} | '
              f'months: {",".join(map(str, months)) if months else "all"} | '
              f'n_models: {n_models or "all"} | threads: {threads} | '
              f'mask_test: {mask_test} | lat_rows: '
              f'{_rows_attr(lat_rows) or "all"}')

    # Check all feature files exist
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

    # --- Process each flux ---
    failures = []
    for flux in selected:
        if flux not in FLUX_METADATA:
            print(f'ERROR: unknown flux {flux}')
            failures.append((flux, 'unknown flux'))
            continue

        models_folder = os.path.join(models_path, flux)
        flux_out = os.path.join(output_path, flux)

        if not os.path.isdir(models_folder):
            print(f'ERROR: models folder not found: {models_folder}')
            failures.append((flux, f'models folder not found: '
                                   f'{models_folder}'))
            continue
        member_names = [f[:-len('.joblib')]
                        for f in _member_files(models_folder, n_models)]
        config = _config_attrs(quantile_method, eff_mask, mask_test,
                               len(member_names), valid_only, out_rows)

        # Check if final output already exists
        final_file = os.path.join(flux_out,
                                  _product_name(flux, year_str, out_rows))
        if os.path.exists(final_file):
            print(f'\n{flux}: final output exists')
            _report_existing(final_file, config)
            if dump_members:
                sfx = _rows_suffix(out_rows)
                ddir = os.path.join(dump_members, flux)
                want = ([os.path.join(ddir, f'{n}{sfx}.nc')
                         for n in member_names] if months is None else
                        [os.path.join(ddir, 'parts', f'{n}_m{m:02d}{sfx}.nc')
                         for n in member_names for m in months])
                gone = [p for p in want if not os.path.exists(p)]
                if gone:
                    msg = (f'product exists but {len(gone)} member dump(s) '
                           f'requested with --dump_members are missing (e.g. '
                           f'{gone[0]}); dumps are only written together with '
                           f'the product: remove {final_file} or use another '
                           f'--output_path')
                    print(f'  ERROR: {msg}')
                    failures.append((flux, msg))
            continue

        if verbose:
            print(f'\n=== {flux}: fused ensemble predict + combine ===')

        fl = _predict_and_combine(
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
            quantile_method=quantile_method,
            months=months,
            valid_only=valid_only,
            dump_members=dump_members,
            n_models=n_models,
            n_jobs=threads,
            mask_test=mask_test,
            lat_rows=out_rows,
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
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Check the quantile combiners against the reference implementation."""

    # Literal transcription of master:src/_percentile.py (keep_good_np,
    # np_percentile), used only as the oracle here.
    def keep_good_np(xx, where, nodata, out=None):
        if out is None:
            out = np.full_like(xx, nodata)
        else:
            assert out.shape == xx.shape
            assert out.dtype == xx.dtype
            assert out is not xx
            out[:] = nodata
        np.copyto(out, xx, where=where)
        return out

    def np_percentile(xx, percentile, nodata):

        if np.isnan(nodata):
            high = True
            mask = ~np.isnan(xx)
        else:
            high = nodata >= xx.max()
            mask = xx != nodata

        valid_counts = mask.sum(axis=0)

        xx = np.sort(xx, axis=0)

        indices = np.round(percentile * (valid_counts - 1))
        if not high:
            indices += xx.shape[0] - valid_counts
            indices[valid_counts == 0] = 0

        indices = indices.astype(np.int64).flatten()
        step = xx.size // xx.shape[0]
        indices = step * indices + np.arange(len(indices))

        xx = xx.take(indices).reshape(xx.shape[1:])

        return keep_good_np(xx, (valid_counts >= 3), nodata)

    def same_bits(a, b):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            return False
        return np.array_equal(a[~na].view(np.uint32), b[~nb].view(np.uint32))

    rng = np.random.default_rng(12345)
    failures = 0
    cases = []
    for n_models, n_pix in [(30, 1000), (15, 1000), (4, 500), (3, 200),
                            (2, 100)]:
        s = rng.normal(5.0, 3.0, size=(n_models, n_pix)).astype('float32')
        s[:, rng.random(n_pix) < 0.1] = np.nan          # whole NaN columns
        # partial NaN columns: random number of invalid members
        part = np.flatnonzero(rng.random(n_pix) < 0.3)
        for c in part:
            k = rng.integers(0, n_models + 1)
            s[rng.choice(n_models, size=k, replace=False), c] = np.nan
        cases.append((f'random {n_models}x{n_pix}', s))
        # ties: discretised values
        t = np.round(rng.normal(0, 2, size=(n_models, n_pix))).astype('float32')
        t[:, rng.random(n_pix) < 0.1] = np.nan
        cases.append((f'ties {n_models}x{n_pix}', t))

    for label, s in cases:
        ours = _nearest_rank_quantiles(s, QUANTILES)
        for k, q in enumerate(QUANTILES):
            ref = np_percentile(s.copy(), q, np.nan)
            ok = same_bits(ours[k], ref)
            failures += not ok
            if not ok:
                print(f'  FAIL nearest {label} q={q}')
        lin = _combine(s, 'linear')
        exp = np.nanquantile(s, list(QUANTILES), axis=0).astype('float32')
        exp[:, np.isfinite(s).sum(axis=0) < 3] = np.nan
        ok = same_bits(lin, exp)
        failures += not ok
        if not ok:
            print(f'  FAIL linear {label}')
        near = _combine(s, 'nearest')
        ok = same_bits(near, ours)
        failures += not ok

    assert _parse_months('1-12') == list(range(1, 13))
    assert _parse_months('1,6,12') == [1, 6, 12]
    assert _parse_months('1-3,7') == [1, 2, 3, 7]
    assert _parse_lat_rows('800:1200') == (800, 1200)
    for bad in ('5', '5:5', '9:3', '-1:4', 'a:b'):
        try:
            _parse_lat_rows(bad)
        except ValueError:
            continue
        failures += 1
        print(f'  FAIL _parse_lat_rows accepted {bad!r}')
    assert _product_name('GPP', 2024, (10, 25)).endswith('_v2.1_r10-25.nc')
    assert _part_name('GPP', 2024, 3, (10, 25)).endswith('_m03_r10-25.nc')
    assert _product_name('GPP', 2024) == _product_name('GPP', 2024, None)
    cfg = _config_attrs('nearest', ['VegH', 'NDWI'], 'nan', 30, True, None)
    assert _config_diff({k: str(v) for k, v in cfg.items()}, cfg) == []
    assert _config_diff({**cfg, 'valid_only': 0}, cfg) == \
        ['valid_only: file=0 requested=1']
    assert _config_diff({}, cfg)[0] == \
        'quantile_method: file=(absent) requested=nearest'
    assert TILE_ROWS % 200 == 0
    for bad in ('0', '13', '', '5-14', 'x'):
        try:
            _parse_months(bad)
        except ValueError:
            continue
        failures += 1
        print(f'  FAIL _parse_months accepted {bad!r}')

    print(f'selftest: {len(cases)} stacks x {len(QUANTILES)} quantiles, '
          f'numpy {np.__version__}: '
          + ('PASS' if failures == 0 else f'{failures} FAILURES'))
    return failures == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 3: fused ensemble prediction + combination.')
    parser.add_argument('--year', type=int, default=None,
                        help='Year to process (e.g. 2024). Required unless '
                             '--selftest.')
    parser.add_argument('--features_path', type=str, default=None,
                        help=f'Feature datasets directory. '
                             f'Default: {DEFAULT_FEATURES} '
                             f'($AUSEFLUX_ROOT/features_500m)')
    parser.add_argument('--static_path', type=str, default=None,
                        help=f'Static input files. Default: {DEFAULT_STATIC} '
                             f'($AUSEFLUX_STATIC)')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Results directory (variables.txt, models, '
                             f'output). Default: {DEFAULT_RESULTS}')
    parser.add_argument('--models_path', type=str, default=None,
                        help='Base folder of ensemble members; models are read '
                             'from {models_path}/{flux}/*.joblib. Default: '
                             '{results}/models/ensemble/v2.1')
    parser.add_argument('--variables_file', type=str, default=None,
                        help='Feature list (variables.txt). Default: '
                             '{results}/variables.txt')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Base output folder; products are written to '
                             '{output_path}/{flux}/. Default: '
                             '{results}/AusEFlux')
    parser.add_argument('--fluxes', type=str, nargs='+',
                        choices=['GPP', 'NEE', 'ER', 'ET'], default=None,
                        help='Fluxes to generate (default: all).')
    parser.add_argument('--quantile_method', choices=['linear', 'nearest'],
                        default='nearest',
                        help='Member combination: nearest (default) = '
                             'reference nearest-rank percentile (np_percentile: '
                             'round-half-even index, >= 3 valid members), '
                             'exact vs the published v2.1 product; linear = '
                             'np.nanquantile, kept for comparison (biased, '
                             '~200x slower).')
    parser.add_argument('--mask_vars', type=str,
                        default=','.join(MASKING_VARS),
                        help='Comma list of features whose NaN marks no-data '
                             f'(default: {",".join(MASKING_VARS)} = reference; '
                             'previous default: VegH,NDWI,rain_anom,Tavg).')
    parser.add_argument('--mask_test', choices=['nonfinite', 'nan'],
                        default='nan',
                        help='What marks a mask-variable pixel as no-data: '
                             'nan (default) = NaN only, like the reference '
                             '(`isnull()`), so +-Inf pixels (e.g. LAI_anom '
                             'where the climatology is 0) are predicted with '
                             'Inf -> 0; nonfinite = NaN or +-Inf (previous '
                             'default).')
    parser.add_argument('--n_models', type=int, default=None,
                        help='Use only the first N >= 1 members (sorted by '
                             'file name). Default: all.')
    parser.add_argument('--months', type=str, default=None,
                        help='Month subset to process, e.g. "1-6" or '
                             '"1,6,12". When given, writes one part file per '
                             'month to {output}/{flux}/parts/..._m{MM}.nc '
                             '(combine with --merge). Default: all months, '
                             'single product file.')
    parser.add_argument('--merge', action='store_true',
                        help='Assemble the 12 month part files into the final '
                             'product (parts are kept) and exit. Parts must '
                             'share the run-configuration attributes.')
    parser.add_argument('--merge_partial', action='store_true',
                        help='Like --merge, but accept fewer than 12 month '
                             'parts (e.g. the current, incomplete year): the '
                             'product time dimension is the number of parts '
                             '(listed in the auseflux_partial_months '
                             'attribute) and a WARNING names the missing '
                             'months.')
    pix = parser.add_mutually_exclusive_group()
    pix.add_argument('--valid_only', dest='valid_only', action='store_true',
                     help='Predict only pixels that are not no-data/urban and '
                          'scatter back (default; bit-identical output, fewer '
                          'tree evaluations).')
    pix.add_argument('--all_pixels', dest='valid_only', action='store_false',
                     help='Predict every pixel then mask (previous default; '
                          'same output, slower).')
    parser.set_defaults(valid_only=True)
    parser.add_argument('--dump_members', type=str, default=None,
                        metavar='DIR',
                        help='Also write each member prediction to '
                             'DIR/{flux}/{member}.nc (variable {flux}, '
                             'float32, NaN where masked); with --months, to '
                             'DIR/{flux}/parts/{member}_m{MM}.nc. Dumps are '
                             'written together with the product: if the '
                             'product already exists but dumps are missing, '
                             'the run fails (remove the product first).')
    parser.add_argument('--threads', type=int, default=PREDICT_N_JOBS,
                        help='n_jobs for member predict and tl2cgen nthread '
                             '(default: -1 = all cores; for tl2cgen, the CPUs '
                             'in the process affinity mask). 0 is invalid.')
    parser.add_argument('--lat_rows', type=str, default=None,
                        metavar='START:END',
                        help='Process only latitude rows START..END-1 of '
                             'the FULL grid (0-based, END exclusive, e.g. '
                             '800:1200). Outputs (product, month parts, member '
                             'dumps) get an _r{START}-{END} suffix and a global '
                             'attribute auseflux_lat_rows. If the features are '
                             'themselves windowed (auseflux_lat_rows '
                             'attribute), START:END must lie inside their '
                             'window; without --lat_rows their window is used. '
                             'With --merge, merges the windowed parts of that '
                             'window. Default: full grid.')
    parser.add_argument('--selftest', action='store_true',
                        help='Run the quantile-combiner self-test and exit.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Progress and timing output (always on).')
    args = parser.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.year is None:
        parser.error('--year is required')
    if (args.merge or args.merge_partial) and args.months:
        parser.error('--merge/--merge_partial cannot be combined with --months')
    if args.n_models is not None and args.n_models < 1:
        parser.error('--n_models must be >= 1')
    if args.threads == 0:
        parser.error('--threads must be -1 (all cores) or >= 1')

    try:
        months = _parse_months(args.months) if args.months else None
        lat_rows = _parse_lat_rows(args.lat_rows) if args.lat_rows else None
    except ValueError as e:
        parser.error(str(e))
    mask_vars = [v.strip() for v in args.mask_vars.split(',') if v.strip()]

    run_ensemble_pipeline(
        year=args.year,
        features_path=args.features_path,
        static_path=args.static_path,
        results_path=args.results_path,
        fluxes=args.fluxes,
        verbose=args.verbose,
        models_path=args.models_path,
        variables_file=args.variables_file,
        output_path=args.output_path,
        mask_vars=mask_vars,
        quantile_method=args.quantile_method,
        months=months,
        merge=args.merge,
        merge_partial=args.merge_partial,
        valid_only=args.valid_only,
        dump_members=args.dump_members,
        n_models=args.n_models,
        threads=args.threads,
        mask_test=args.mask_test,
        lat_rows=lat_rows,
    )


if __name__ == '__main__':
    main()
