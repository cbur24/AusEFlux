"""
Quantile-distillation training (Level 5, faithful to the pipeline README).

For each flux:
  1. sample valid feature pixels across several years / all months,
  2. run every one of the 30 pre-trained ensemble models on the sample,
  3. take the empirical 25th / 50th / 75th percentiles across the 30
     predictions per sample as the training targets (same np.nanquantile
     definition the ensemble-combination step uses),
  4. fit 3 quantile-loss LightGBM models (alpha = 0.25 / 0.50 / 0.75) and
     save them where step3_ensemble_prediction_quantile.py --mode predict
     expects them:  {results}/models/quantile/v2.1/{flux}/{flux}_q{25,50,75}.joblib

Run in an env matching the reference models (RandomForest needs sklearn 1.3.2):
    uv run --python 3.12 --with "scikit-learn==1.3.2" --with "numpy<2" \
        python distill_quantile_models.py --fluxes GPP ER NEE ET
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "numpy<2", "pandas", "xarray>=2025.1.0", "netCDF4",
#     "joblib", "lightgbm", "scikit-learn==1.3.2",
# ]
# ///

import argparse
import gc
import os
import sys

import joblib
import numpy as np
import xarray as xr

BASE = "/home/prl900/Downloads/wald_eflux"
FEATURES = os.path.join(BASE, "features_500m")
RESULTS = os.path.join(BASE, "results")
TARGET_GRID = "500m"
VERSION = "v2.1"
MASKING_VARS = ["VegH", "NDWI", "rain_anom", "Tavg"]
QUANTILES = {"q25": 0.25, "q50": 0.50, "q75": 0.75}
FLUXES = ["GPP", "ER", "NEE", "ET"]


def _data_var(ds):
    skip = {"crs", "spatial_ref"}
    return next(v for v in ds.data_vars if v not in skip and ds[v].ndim > 0)


def _parse_variables(path):
    with open(path) as f:
        header = f.read().strip().split(",")
    return [v[:-3] for v in header if v and v != "site"]


def build_sample(train_vars, years, n_per_year, seed=42, features_dir=None):
    """Assemble a (N, n_vars) feature sample over valid pixels of `years`.

    Memory-safe at continental scale: instead of materialising the full
    (12*lat*lon, n_vars) stack (~68 GB for a continental year), it determines the
    valid-pixel sample indices from the masking variables first, then gathers only
    the sampled rows one variable at a time (peak ~one variable, ~2.7 GB).
    """
    features_dir = features_dir or FEATURES
    rng = np.random.default_rng(seed)
    n_vars = len(train_vars)
    das = {}
    for v in train_vars:
        ds = xr.open_dataset(os.path.join(features_dir, f"{v}_{TARGET_GRID}.nc"))
        das[v] = ds[_data_var(ds)]
    chunks = []
    for y in years:
        # valid = all masking vars finite (load one masking var at a time)
        valid = None
        for v in MASKING_VARS:
            if v not in train_vars:
                continue
            a = das[v].sel(time=str(y)).values.ravel()
            f = np.isfinite(a)
            valid = f if valid is None else (valid & f)
            del a, f
        idx = np.where(valid)[0]
        del valid
        if idx.size > n_per_year:
            idx = rng.choice(idx, n_per_year, replace=False)
        idx.sort()
        # gather sampled rows per variable
        Xy = np.empty((idx.size, n_vars), dtype="float32")
        for j, v in enumerate(train_vars):
            a = das[v].sel(time=str(y)).values.ravel()
            Xy[:, j] = a[idx]
            del a
        Xy[~np.isfinite(Xy)] = 0.0  # clean as step3 does
        chunks.append(Xy)
        print(f"    year {y}: {idx.size} samples")
    for ds in das.values():
        ds.close()
    return np.concatenate(chunks, axis=0)


def ensemble_quantile_targets(flux, X):
    """Run 30 ensemble models on X, return (3, N) of 25/50/75 percentiles."""
    mfolder = os.path.join(RESULTS, "models", "ensemble", VERSION, flux)
    model_files = sorted(f for f in os.listdir(mfolder) if f.endswith(".joblib"))
    preds = np.empty((len(model_files), X.shape[0]), dtype="float32")
    for i, mf in enumerate(model_files):
        m = joblib.load(os.path.join(mfolder, mf))
        m.set_params(n_jobs=-1)
        preds[i] = m.predict(X).astype("float32")
        del m
        gc.collect()
        print(f"    ensemble {i+1}/{len(model_files)} ({mf})", end="\r")
    print()
    q = np.nanquantile(preds, [0.25, 0.50, 0.75], axis=0)  # linear, matches combine
    del preds
    return q


def train_and_save(flux, X, targets):
    from lightgbm import LGBMRegressor
    mfolder = os.path.join(RESULTS, "models", "quantile", VERSION, flux)
    os.makedirs(mfolder, exist_ok=True)
    for (qname, alpha), tgt in zip(QUANTILES.items(), targets):
        model = LGBMRegressor(
            objective="quantile", alpha=alpha, boosting_type="gbdt",
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            min_child_samples=30, max_depth=-1, random_state=42, n_jobs=-1,
            verbose=-1,
        )
        model.fit(X, tgt.astype("float32"))
        out = os.path.join(mfolder, f"{flux}_{qname}.joblib")
        joblib.dump(model, out)
        # quick in-sample fit check
        pred = model.predict(X)
        mae = float(np.mean(np.abs(pred - tgt)))
        print(f"    {flux}_{qname} (alpha={alpha}) trained, in-sample MAE={mae:.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fluxes", nargs="+", default=FLUXES)
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2004, 2009, 2014, 2019, 2020])
    ap.add_argument("--n-per-year", type=int, default=80000)
    ap.add_argument("--features-dir", default=None,
                    help="feature dir to sample from (default bbox features_500m)")
    args = ap.parse_args()

    vars_path = os.path.join(RESULTS, "variables.txt")
    train_vars = _parse_variables(vars_path)
    print(f"features ({len(train_vars)}): {', '.join(train_vars)}")
    print(f"sampling years {args.years}, {args.n_per_year}/year")

    print("\nBuilding feature sample (shared across fluxes)...")
    X = build_sample(train_vars, args.years, args.n_per_year,
                     features_dir=args.features_dir)
    print(f"total sample: {X.shape}")

    for flux in args.fluxes:
        print(f"\n=== {flux}: distilling ===")
        targets = ensemble_quantile_targets(flux, X)
        train_and_save(flux, X, targets)
        del targets
        gc.collect()
    print("\nDistillation complete.")


if __name__ == "__main__":
    main()
