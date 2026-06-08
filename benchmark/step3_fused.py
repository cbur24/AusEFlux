"""
Fused ensemble predict + combine (efficiency variant of step3).

The shipped `step3_ensemble_prediction.py` loops `for model (30): for time: for
band: read X; predict; write per-model file`, then a separate combine step reads
those 30 files back to compute quantiles. That reads every feature 30× and does a
30-file write+read round-trip.

This variant inverts the loops: `for time: for band: read X ONCE; predict all 30
models; compute 25/50/75 percentiles on the tile; write the final product
directly`. Because the no-data and urban masks are identical across models, the
per-tile quantile over the 30 predictions is mathematically identical to the
shipped path — only the I/O changes. Output goes to results/AusEFlux_fused/.

    uv run step3_fused.py --flux GPP --year 2020 \
        --features_path .../features_500m --static_path .../input/static
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2", "pandas", "xarray>=2025.1.0", "netCDF4",
#                 "joblib", "lightgbm", "scikit-learn==1.3.2"]
# ///
import argparse
import gc
import os
import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")
import joblib
import netCDF4 as nc4
import numpy as np
import pandas as pd
import xarray as xr

TARGET_GRID = "500m"
VERSION = "v2.1"
MASKING_VARS = ["VegH", "NDWI", "rain_anom", "Tavg"]
FLUX_METADATA = {
    "GPP": {"long_name": "Gross Primary Productivity", "units": "gC/m\N{SUPERSCRIPT TWO}/month"},
    "NEE": {"long_name": "Net Ecosystem Exchange", "units": "gC/m\N{SUPERSCRIPT TWO}/month"},
    "ER":  {"long_name": "Ecosystem Respiration", "units": "gC/m\N{SUPERSCRIPT TWO}/month"},
    "ET":  {"long_name": "Evapotranspiration", "units": "mm/month"},
}


def _data_var(ds):
    skip = {"crs", "spatial_ref"}
    return next(v for v in ds.data_vars if v not in skip and ds[v].ndim > 0)


def _coords(da):
    lat = np.array([round(float(v), 4) for v in da.latitude.values], dtype="float32")
    lon = np.array([round(float(v), 4) for v in da.longitude.values], dtype="float32")
    return lat, lon


def _parse_variables(path):
    with open(path) as f:
        header = f.read().strip().split(",")
    return [v[:-3] for v in header if v and v != "site"]


@contextmanager
def _nc_full(path, var_names, lat, lon, times, attrs=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds = nc4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("time", len(times))
        ds.createDimension("latitude", len(lat))
        ds.createDimension("longitude", len(lon))
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "days since 1970-01-01"; tv.calendar = "proleptic_gregorian"
        epoch = pd.Timestamp("1970-01-01")
        for i, t in enumerate(times):
            tv[i] = float((pd.Timestamp(t) - epoch).days)
        lv = ds.createVariable("latitude", "f4", ("latitude",)); lv.units = "degrees_north"; lv[:] = lat
        xv = ds.createVariable("longitude", "f4", ("longitude",)); xv.units = "degrees_east"; xv[:] = lon
        for vn in var_names:
            ds.createVariable(vn, "f4", ("time", "latitude", "longitude"),
                              fill_value=np.float32(np.nan), zlib=True, complevel=4,
                              chunksizes=(1, min(200, len(lat)), min(200, len(lon))))
        for k, v in (attrs or {}).items():
            ds.setncattr(k, v)
        yield ds
    finally:
        ds.close()


def run(flux, year, features_path, static_path, results_path, tile_rows,
        n_models_keep=None, out_subdir="AusEFlux_fused", verbose=True):
    year = str(year)
    train_vars = _parse_variables(os.path.join(results_path, "variables.txt"))
    n_vars = len(train_vars)
    mask_idx = [train_vars.index(v) for v in MASKING_VARS if v in train_vars]

    # grid + timestamps from first feature
    ref = xr.open_dataset(os.path.join(features_path, f"{train_vars[0]}_{TARGET_GRID}.nc"))
    ref_da = ref[_data_var(ref)].sel(time=year)
    lat, lon = _coords(ref_da)
    n_lat, n_lon = len(lat), len(lon)
    timestamps = [pd.Timestamp(t) for t in ref_da.time.values]
    n_time = len(timestamps)
    ref.close()

    # urban/water mask
    up = os.path.join(static_path, f"urban_water_mask_{TARGET_GRID}.nc")
    if os.path.exists(up):
        uds = xr.open_dataset(up)
        urban = (uds[_data_var(uds)].values == 1)
        uds.close()
    else:
        urban = np.zeros((n_lat, n_lon), dtype=bool)

    # feature datasets (lazy) + models
    feats = {}
    for v in train_vars:
        ds = xr.open_dataset(os.path.join(features_path, f"{v}_{TARGET_GRID}.nc"))
        feats[v] = ds[_data_var(ds)].sel(time=year)
    mfolder = os.path.join(results_path, "models", "ensemble", VERSION, flux)
    model_files = sorted(f for f in os.listdir(mfolder) if f.endswith(".joblib"))
    if n_models_keep:
        # interleave lgbm/rf so any N keeps a mix, then take the first N
        lgbm = [f for f in model_files if "lgbm" in f]
        rf = [f for f in model_files if "rf" in f]
        inter = [f for pair in zip(lgbm, rf) for f in pair]
        inter += [f for f in model_files if f not in inter]  # any leftovers
        model_files = inter[:n_models_keep]
    models = []
    for mf in model_files:
        m = joblib.load(os.path.join(mfolder, mf))
        try:
            m.set_params(n_jobs=-1)
        except Exception:
            pass
        models.append(m)
    n_models = len(models)
    if verbose:
        print(f"{flux}: {n_models} models, grid {n_lat}x{n_lon}, {n_time} steps")

    var_names = [f"{flux}_25th_percentile", f"{flux}_median", f"{flux}_75th_percentile"]
    meta = FLUX_METADATA[flux]
    attrs = {"crs": "EPSG:4326", "short_name": flux, "long_name": meta["long_name"],
             "units": meta["units"], "version": VERSION,
             "description": (f"AusEFlux {meta['long_name']} is created by empirically "
                             "upscaling the OzFlux eddy covariance network using machine "
                             "learning methods coupled with climate and remote sensing "
                             "datasets. The estimates provided within this dataset were "
                             "extracted from an ensemble of predictions and represent the "
                             "median and uncertainty range.")}
    out_file = os.path.join(results_path, out_subdir, flux,
                            f"AusEFlux_{flux}_{TARGET_GRID}_quantiles_{year}_{VERSION}.nc")
    if os.path.exists(out_file):
        os.remove(out_file)

    t0 = time.time()
    with _nc_full(out_file, var_names, lat, lon, timestamps, attrs=attrs) as nc_out:
        for t in range(n_time):
            if verbose:
                print(f"    time {t+1:02d}/{n_time}", end="\r")
            for lat_s in range(0, n_lat, tile_rows):
                lat_e = min(lat_s + tile_rows, n_lat)
                tile_h = lat_e - lat_s
                n_pix = tile_h * n_lon

                # --- read features ONCE for this tile ---
                X = np.empty((n_pix, n_vars), dtype="float32")
                for v, var in enumerate(train_vars):
                    X[:, v] = feats[var].isel(
                        time=t, latitude=slice(lat_s, lat_e)).values.ravel().astype("float32")
                nodata = np.zeros(n_pix, dtype=bool)
                for vi in mask_idx:
                    nodata |= ~np.isfinite(X[:, vi])
                X[~np.isfinite(X)] = 0.0

                # --- predict all models on the same X ---
                stack = np.empty((n_models, n_pix), dtype="float32")
                for mi, m in enumerate(models):
                    stack[mi] = m.predict(X).astype("float32")
                stack[:, nodata] = np.nan
                # urban mask (flatten this band)
                ub = urban[lat_s:lat_e, :].ravel()
                stack[:, ub] = np.nan

                valid = np.sum(np.isfinite(stack), axis=0)
                q = np.nanquantile(stack, [0.25, 0.50, 0.75], axis=0).astype("float32")
                insufficient = valid < 3
                q[:, insufficient] = np.nan

                for b in range(3):
                    nc_out[var_names[b]][t, lat_s:lat_e, :] = q[b].reshape(tile_h, n_lon)
                del X, stack, q
                gc.collect()
    if verbose:
        print()
    print(f"{flux}: fused predict+combine done in {time.time()-t0:.1f}s -> {out_file}")
    for da in feats.values():
        da.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flux", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--features_path", default="/home/prl900/Downloads/wald_eflux/features_500m")
    ap.add_argument("--static_path", default="/home/prl900/Downloads/wald_eflux/input/static")
    ap.add_argument("--results_path", default="/home/prl900/Downloads/wald_eflux/results")
    ap.add_argument("--tile-rows", type=int, default=1500)
    ap.add_argument("--n-models", type=int, default=None,
                    help="use only the first N ensemble members (interleaved lgbm/rf)")
    ap.add_argument("--out-subdir", default="AusEFlux_fused",
                    help="output subdir under results_path")
    args = ap.parse_args()
    run(args.flux, args.year, args.features_path, args.static_path,
        args.results_path, args.tile_rows, n_models_keep=args.n_models,
        out_subdir=args.out_subdir)


if __name__ == "__main__":
    main()
