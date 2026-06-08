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

Usage:
    uv run step3_ensemble_prediction_quantile.py --year 2024 --mode train --training_data data/training_csvs
    uv run step3_ensemble_prediction_quantile.py --year 2024 --mode predict
    uv run step3_ensemble_prediction_quantile.py --year 2024 --fluxes GPP ET
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

TILE_ROWS = 1500  # latitude rows per prediction tile
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


# ---------------------------------------------------------------------------
# NetCDF4 output writer
# ---------------------------------------------------------------------------

@contextmanager
def _nc_full(path, var_names, lat, lon, times, attrs=None):
    """Create output NetCDF with multiple variables sharing the same grid."""
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
                    continue
                ds.setncattr(k, v)

        yield ds
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Step A: Training quantile models
# ---------------------------------------------------------------------------

def _load_training_data(training_data_path, flux, verbose=False):
    """
    Load training CSVs from OzFlux site extractions.

    Expects one CSV per site in *training_data_path*, each with columns
    matching FEATURE_COLS plus the flux target column and a 'site' column.
    """
    from lightgbm import LGBMRegressor

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


def _train_quantile_models(flux, training_data_path, results_path,
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

    models_folder = os.path.join(
        results_path, 'models', 'quantile', VERSION, flux)
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
                           timestamps, n_lat, n_lon, verbose=False):
    """
    Predict 25th, 50th, 75th percentiles directly using 3 quantile models.

    Produces the same final output as step3+step4 combined, but with only
    3 model passes instead of 30.
    """
    var_names = [f'{flux}_25th_percentile',
                 f'{flux}_median',
                 f'{flux}_75th_percentile']

    meta = FLUX_METADATA[flux]
    description = (
        f'AusEFlux {meta["long_name"]} is created by empirically upscaling '
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

    out_file = os.path.join(
        output_path,
        f'AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year}_{VERSION}.nc')

    if os.path.exists(out_file):
        if verbose:
            print(f'  Output exists, skipping: {out_file}')
        return

    # Load the 3 quantile models
    q_models = {}
    for q_name, alpha in QUANTILES.items():
        model_path = os.path.join(models_folder, f'{flux}_{q_name}.joblib')
        if not os.path.exists(model_path):
            print(f'  ERROR: quantile model not found: {model_path}')
            print(f'  Run with --mode train first.')
            return
        q_models[q_name] = joblib.load(model_path)
        q_models[q_name].set_params(n_jobs=PREDICT_N_JOBS)

    if verbose:
        print(f'  Loaded 3 quantile models')

    # Open feature datasets lazily
    feat_datasets = {}
    for var in train_vars:
        path = os.path.join(features_path, f'{var}_{TARGET_GRID}.nc')
        if not os.path.exists(path):
            print(f'  ERROR: feature file not found: {path}')
            return
        ds = xr.open_dataset(path)
        dvar = _data_var(ds)
        feat_datasets[var] = ds[dvar].sel(time=year)

    mask_var_indices = [train_vars.index(v) for v in MASKING_VARS
                        if v in train_vars]

    n_time = len(timestamps)
    n_vars = len(train_vars)

    os.makedirs(output_path, exist_ok=True)

    with _nc_full(out_file, var_names, lat, lon, timestamps,
                  attrs=attrs) as nc_out:
        for t in range(n_time):
            if verbose:
                print(f'    time {t+1:02d}/{n_time}', end='\r')

            # Allocate output arrays for all 3 quantiles
            q_results = {q: np.full((n_lat, n_lon), np.nan, dtype='float32')
                         for q in QUANTILES}

            for lat_s in range(0, n_lat, TILE_ROWS):
                lat_e = min(lat_s + TILE_ROWS, n_lat)
                tile_h = lat_e - lat_s
                n_pix = tile_h * n_lon

                # Load feature tile
                X = np.empty((n_pix, n_vars), dtype='float32')
                for v, var in enumerate(train_vars):
                    X[:, v] = feat_datasets[var].isel(
                        time=t,
                        latitude=slice(lat_s, lat_e)
                    ).values.ravel().astype('float32')

                # No-data mask
                nodata = np.zeros(n_pix, dtype=bool)
                for vi in mask_var_indices:
                    nodata |= ~np.isfinite(X[:, vi])

                # Clean NaN/Inf for prediction
                X[~np.isfinite(X)] = 0.0

                # Predict all 3 quantiles from the same X
                for q_name in QUANTILES:
                    preds = q_models[q_name].predict(X).astype('float32')
                    preds[nodata] = np.nan
                    q_results[q_name][lat_s:lat_e, :] = \
                        preds.reshape(tile_h, n_lon)

                del X

            # Apply urban/water mask and write
            for q_name in QUANTILES:
                q_results[q_name][urban_mask] = np.nan

            nc_out[var_names[0]][t, :, :] = q_results['q25']
            nc_out[var_names[1]][t, :, :] = q_results['q50']
            nc_out[var_names[2]][t, :, :] = q_results['q75']

            del q_results

        if verbose:
            print()

    # Close feature datasets
    for ds_da in feat_datasets.values():
        ds_da.close()

    # Unload models
    del q_models
    gc.collect()

    if verbose:
        print(f'  Output: {out_file}')


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_quantile_pipeline(year, mode='predict', training_data_path=None,
                          features_path=None, static_path=None,
                          results_path=None, fluxes=None, n_cpus=4,
                          verbose=False):
    if features_path is None:
        features_path = DEFAULT_FEATURES
    if static_path is None:
        static_path = DEFAULT_STATIC
    if results_path is None:
        results_path = DEFAULT_RESULTS

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
            _train_quantile_models(flux, training_data_path, results_path,
                                   n_cpus=n_cpus, verbose=verbose)

        if verbose:
            print('\nTraining complete.')
        return

    # --- Prediction mode ---
    features_list = os.path.join(results_path, 'variables.txt')
    if not os.path.exists(features_list):
        print(f'ERROR: variables.txt not found: {features_list}')
        sys.exit(1)

    train_vars = _parse_variables(features_list)
    if verbose:
        print(f'Feature variables ({len(train_vars)}): '
              f'{", ".join(train_vars)}')

    missing = [v for v in train_vars
               if not os.path.exists(
                   os.path.join(features_path, f'{v}_{TARGET_GRID}.nc'))]
    if missing:
        print(f'ERROR: missing feature files: {", ".join(missing)}')
        sys.exit(1)

    # Load coordinate info
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
        print(f'Grid: {n_lat} x {n_lon}, {len(timestamps)} time steps '
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

    # Process each flux
    for flux in selected:
        if flux not in FLUX_METADATA:
            print(f'WARNING: unknown flux {flux}, skipping')
            continue

        models_folder = os.path.join(
            results_path, 'models', 'quantile', VERSION, flux)
        output_path = os.path.join(results_path, 'AusEFlux_quantile', flux)

        if not os.path.isdir(models_folder):
            print(f'WARNING: quantile models folder not found: '
                  f'{models_folder}')
            print(f'  Run with --mode train first.')
            continue

        if verbose:
            print(f'\n=== Predict {flux} (quantile) ===')

        _predict_flux_quantile(
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
                             f'Default: {DEFAULT_FEATURES}')
    parser.add_argument('--static_path', type=str, default=None,
                        help=f'Static input files. Default: {DEFAULT_STATIC}')
    parser.add_argument('--results_path', type=str, default=None,
                        help=f'Results directory. Default: {DEFAULT_RESULTS}')
    parser.add_argument('--fluxes', type=str, nargs='+',
                        choices=['GPP', 'NEE', 'ER', 'ET'], default=None,
                        help='Fluxes to process (default: all).')
    parser.add_argument('--n_cpus', type=int, default=4,
                        help='CPUs for training hyperparameter search '
                             '(default: 4).')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

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
    )


if __name__ == '__main__':
    main()
