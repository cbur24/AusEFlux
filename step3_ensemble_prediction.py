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

Usage:
    uv run step3_ensemble_prediction.py --year 2024
    uv run step3_ensemble_prediction.py --year 2024 --fluxes GPP ET
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
import os
import sys
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

BASE_PATH = '/home/prl900/Downloads/wald_eflux'
DEFAULT_FEATURES = os.path.join(BASE_PATH, 'features_500m')
DEFAULT_STATIC   = os.path.join(BASE_PATH, 'input', 'static')
DEFAULT_RESULTS  = os.path.join(BASE_PATH, 'results')
TARGET_GRID = '500m'
VERSION = 'v2.1'

MASKING_VARS = ['VegH', 'NDWI', 'rain_anom', 'Tavg']

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

TILE_ROWS = 1500  # latitude rows per prediction tile
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


# ---------------------------------------------------------------------------
# NetCDF4 output writer (full dimensions known upfront)
# ---------------------------------------------------------------------------

@contextmanager
def _nc_full(path, var_names, lat, lon, times, attrs=None):
    """
    Create output NetCDF with multiple variables sharing the same grid.
    *var_names* is a list of variable names to create.
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

        for vn in var_names:
            ds.createVariable(vn, 'f4', ('time', 'latitude', 'longitude'),
                              fill_value=np.float32(np.nan), zlib=True,
                              complevel=4,
                              chunksizes=(1, min(200, len(lat)),
                                          min(200, len(lon))))

        if attrs:
            for k, v in attrs.items():
                if isinstance(v, float) and np.isnan(v):
                    continue  # netCDF4 doesn't like NaN attrs
                ds.setncattr(k, v)

        yield ds
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Ensemble member predictors (joblib, with optional treelite-compiled LightGBM)
# ---------------------------------------------------------------------------

def _load_predictors(models_folder, verbose=False):
    """
    Return a list of `predict(X)->1d float32` callables, one per ensemble member.

    For each member, a compiled `compiled/<member>.so` (treelite/tl2cgen) is used
    if present (LightGBM only, exact, ~1.5x faster); otherwise the joblib model is
    loaded.  RandomForest members are always joblib (treelite does not help them).
    """
    model_files = sorted(f for f in os.listdir(models_folder)
                         if f.endswith('.joblib'))
    compiled_dir = os.path.join(models_folder, 'compiled')
    predictors, n_compiled = [], 0
    for mf in model_files:
        so = os.path.join(compiled_dir, mf[:-len('.joblib')] + '.so')
        if os.path.exists(so):
            import tl2cgen  # lazy: only needed when compiled members exist
            pr = tl2cgen.Predictor(so, nthread=os.cpu_count())
            predictors.append(
                lambda X, pr=pr: pr.predict(tl2cgen.DMatrix(X)).ravel().astype('float32'))
            n_compiled += 1
        else:
            m = joblib.load(os.path.join(models_folder, mf))
            try:
                m.set_params(n_jobs=PREDICT_N_JOBS)
            except Exception:
                pass
            predictors.append(lambda X, m=m: m.predict(X).astype('float32'))
    if verbose:
        extra = f' ({n_compiled} treelite-compiled)' if n_compiled else ''
        print(f'  {len(predictors)} ensemble members{extra}')
    return predictors


# ---------------------------------------------------------------------------
# Fused predict + combine
# ---------------------------------------------------------------------------

def _predict_and_combine(flux, year, models_folder, features_path, train_vars,
                         output_path, urban_mask, lat, lon, timestamps,
                         n_lat, n_lon, verbose=False):
    """
    Predict all ensemble members and write the 25/50/75-percentile product in a
    single pass: features per (time, latitude-band) tile are read once, every
    member predicts on them, and the per-tile quantiles are written directly.
    """
    predictors = _load_predictors(models_folder, verbose=verbose)
    n_models = len(predictors)
    if n_models == 0:
        print(f'  WARNING: no models found in {models_folder}')
        return

    out_file = os.path.join(
        output_path,
        f'AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year}_{VERSION}.nc')

    # Open feature datasets lazily, selecting the target year.
    feat_datasets = {}
    for var in train_vars:
        path = os.path.join(features_path, f'{var}_{TARGET_GRID}.nc')
        if not os.path.exists(path):
            print(f'  ERROR: feature file not found: {path}')
            return
        ds = xr.open_dataset(path)
        feat_datasets[var] = ds[_data_var(ds)].sel(time=year)

    mask_var_indices = [train_vars.index(v) for v in MASKING_VARS
                        if v in train_vars]
    n_time = len(timestamps)
    n_vars = len(train_vars)

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
            f'AusEFlux {meta["long_name"]} is created by empirically upscaling '
            'the OzFlux eddy covariance network using machine learning methods '
            'coupled with climate and remote sensing datasets. The estimates '
            'provided within this dataset were extracted from an ensemble of '
            'predictions and represent the median and uncertainty range.'),
    }

    os.makedirs(output_path, exist_ok=True)
    with _nc_full(out_file, var_names, lat, lon, timestamps, attrs=attrs) as nc_out:
        for t in range(n_time):
            if verbose:
                print(f'    time {t+1:02d}/{n_time}', end='\r')

            for lat_s in range(0, n_lat, TILE_ROWS):
                lat_e = min(lat_s + TILE_ROWS, n_lat)
                tile_h = lat_e - lat_s
                n_pix = tile_h * n_lon

                # --- read features ONCE for this tile ---
                X = np.empty((n_pix, n_vars), dtype='float32')
                for v, var in enumerate(train_vars):
                    X[:, v] = feat_datasets[var].isel(
                        time=t, latitude=slice(lat_s, lat_e)
                    ).values.ravel().astype('float32')

                # No-data mask: any masking var is NaN
                nodata = np.zeros(n_pix, dtype=bool)
                for vi in mask_var_indices:
                    nodata |= ~np.isfinite(X[:, vi])
                X[~np.isfinite(X)] = 0.0  # clean NaN/Inf -> 0 for prediction

                # --- predict every member on the same X ---
                stack = np.empty((n_models, n_pix), dtype='float32')
                for mi, predict in enumerate(predictors):
                    stack[mi] = predict(X)
                stack[:, nodata] = np.nan
                stack[:, urban_mask[lat_s:lat_e, :].ravel()] = np.nan

                # --- combine: 25/50/75 percentiles (>= 3 valid members) ---
                valid_count = np.sum(np.isfinite(stack), axis=0)
                q = np.nanquantile(stack, [0.25, 0.50, 0.75],
                                   axis=0).astype('float32')
                q[:, valid_count < 3] = np.nan

                for b in range(3):
                    nc_out[var_names[b]][t, lat_s:lat_e, :] = \
                        q[b].reshape(tile_h, n_lon)
                del X, stack, q
                gc.collect()

    if verbose:
        print()
        print(f'  Output: {out_file}')

    for da in feat_datasets.values():
        da.close()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_ensemble_pipeline(year, features_path=None, static_path=None,
                          results_path=None, fluxes=None, verbose=False):
    """
    Run the fused ensemble predict+combine for all requested fluxes.
    """
    if features_path is None:
        features_path = DEFAULT_FEATURES
    if static_path is None:
        static_path = DEFAULT_STATIC
    if results_path is None:
        results_path = DEFAULT_RESULTS

    year_str = str(year)

    # --- Check inputs ---
    features_list = os.path.join(results_path, 'variables.txt')
    if not os.path.exists(features_list):
        print(f'ERROR: variables.txt not found: {features_list}')
        sys.exit(1)

    train_vars = _parse_variables(features_list)
    if verbose:
        print(f'Feature variables ({len(train_vars)}): {", ".join(train_vars)}')

    # Check all feature files exist
    missing = [v for v in train_vars
               if not os.path.exists(
                   os.path.join(features_path, f'{v}_{TARGET_GRID}.nc'))]
    if missing:
        print(f'ERROR: missing feature files: {", ".join(missing)}')
        sys.exit(1)

    # Load coordinate info from first feature file
    ref_path = os.path.join(features_path,
                            f'{train_vars[0]}_{TARGET_GRID}.nc')
    ref_ds = xr.open_dataset(ref_path)
    ref_da = ref_ds[_data_var(ref_ds)]
    ref_da_year = ref_da.sel(time=year_str)
    lat, lon = _coords(ref_da_year)
    n_lat = len(lat)
    n_lon = len(lon)
    timestamps = [pd.Timestamp(t) for t in ref_da_year.time.values]
    ref_ds.close()

    if verbose:
        print(f'Grid: {n_lat} × {n_lon}, {len(timestamps)} time steps '
              f'for year {year_str}')

    # Load urban/water mask
    urban_path = os.path.join(static_path,
                              f'urban_water_mask_{TARGET_GRID}.nc')
    if os.path.exists(urban_path):
        urban_ds = xr.open_dataset(urban_path)
        urban_var = _data_var(urban_ds)
        urban_mask = (urban_ds[urban_var].values == 1)
        urban_ds.close()
    else:
        print(f'  WARNING: urban/water mask not found ({urban_path}), '
              f'proceeding without it')
        urban_mask = np.zeros((n_lat, n_lon), dtype=bool)

    # --- Process each flux ---
    selected = fluxes if fluxes else list(FLUX_METADATA.keys())

    for flux in selected:
        if flux not in FLUX_METADATA:
            print(f'WARNING: unknown flux {flux}, skipping')
            continue

        models_folder = os.path.join(
            results_path, 'models', 'ensemble', VERSION, flux)
        output_path = os.path.join(results_path, 'AusEFlux', flux)

        if not os.path.isdir(models_folder):
            print(f'WARNING: models folder not found: {models_folder}')
            continue

        # Check if final output already exists
        final_file = os.path.join(
            output_path,
            f'AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year_str}_{VERSION}.nc')
        if os.path.exists(final_file):
            if verbose:
                print(f'\n{flux}: final output exists, skipping ({final_file})')
            continue

        if verbose:
            print(f'\n=== {flux}: fused ensemble predict + combine ===')

        _predict_and_combine(
            flux=flux,
            year=year_str,
            models_folder=models_folder,
            features_path=features_path,
            train_vars=train_vars,
            output_path=output_path,
            urban_mask=urban_mask,
            lat=lat, lon=lon,
            timestamps=timestamps,
            n_lat=n_lat, n_lon=n_lon,
            verbose=verbose,
        )
        gc.collect()

    if verbose:
        print('\nAll fluxes complete.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AusEFlux Step 3: fused ensemble prediction + combination.')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to process (e.g. 2024).')
    parser.add_argument('--features_path', type=str, default=None,
                        help=f'Feature datasets directory. '
                             f'Default: {DEFAULT_FEATURES}')
    parser.add_argument('--static_path', type=str, default=None,
                        help=f'Static input files. Default: {DEFAULT_STATIC}')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Results directory (models + output). '
                             f'Default: {DEFAULT_RESULTS}')
    parser.add_argument('--fluxes', type=str, nargs='+',
                        choices=['GPP', 'NEE', 'ER', 'ET'], default=None,
                        help='Fluxes to generate (default: all).')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    run_ensemble_pipeline(
        year=args.year,
        features_path=args.features_path,
        static_path=args.static_path,
        results_path=args.results_path,
        fluxes=args.fluxes,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
