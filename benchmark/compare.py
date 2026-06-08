"""
Reusable comparison of two gridded NetCDF variables (new pipeline vs. reference).

Aligns the two arrays on a common grid (nearest, within a small tolerance),
restricts to pixels valid in BOTH, and reports per-pixel agreement metrics plus
optional diagnostic plots (difference map, density scatter, residual histogram,
regional monthly time series).

Run inside the local uv project env:
    uv run python compare.py NEW.nc REF.nc --new-var GPP_median --ref-var GPP_median
    uv run python compare.py NEW.nc REF.nc --new-var GPP_median --ref-var GPP_median --plots out/
"""

import argparse
import json
import os

import numpy as np
import xarray as xr


def _coord_names(da):
    lat = next((c for c in ("latitude", "lat", "y") if c in da.coords or c in da.dims), None)
    lon = next((c for c in ("longitude", "lon", "x") if c in da.coords or c in da.dims), None)
    return lat, lon


def _open_var(path, var):
    ds = xr.open_dataset(path)
    if var is None:
        skip = {"crs", "spatial_ref"}
        var = next(v for v in ds.data_vars if v not in skip and ds[v].ndim >= 2)
    da = ds[var]
    lat, lon = _coord_names(da)
    return da.rename({lat: "lat", lon: "lon"}), var


def align(new, ref):
    """Align ref onto new's grid by nearest-neighbour (grids are nominally equal)."""
    # Order lat ascending for both to avoid slice surprises.
    for d in ("lat", "lon"):
        if bool(new[d][0] > new[d][-1]):
            new = new.sortby(d)
        if bool(ref[d][0] > ref[d][-1]):
            ref = ref.sortby(d)
    # Crop to the overlapping extent so we only compare where both have a grid
    # (avoids counting one product's larger footprint as a mask mismatch).
    lo_lat = max(float(new["lat"].min()), float(ref["lat"].min()))
    hi_lat = min(float(new["lat"].max()), float(ref["lat"].max()))
    lo_lon = max(float(new["lon"].min()), float(ref["lon"].min()))
    hi_lon = min(float(new["lon"].max()), float(ref["lon"].max()))
    new = new.sel(lat=slice(lo_lat, hi_lat), lon=slice(lo_lon, hi_lon))
    ref_i = ref.interp(lat=new["lat"], lon=new["lon"], method="nearest")
    return new, ref_i


def metrics(a, b):
    """a = new, b = ref; both 1-D arrays over pixels valid in both."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out = {"n_pixels": int(m.sum())}
    if a.size == 0:
        return out
    diff = a - b
    rng = float(np.nanmax(b) - np.nanmin(b)) or 1.0
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2)) or np.nan
    out.update(
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        mae=float(np.mean(np.abs(diff))),
        mean_bias=float(np.mean(diff)),
        max_abs_diff=float(np.max(np.abs(diff))),
        r2=float(1 - ss_res / ss_tot) if np.isfinite(ss_tot) else None,
        pearson=float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else None,
        pct_within_1pct_range=float(np.mean(np.abs(diff) < 0.01 * rng) * 100),
        ref_range=rng,
        new_mean=float(a.mean()), ref_mean=float(b.mean()),
    )
    return out


def mask_jaccard(new, ref):
    """Agreement of valid-pixel masks (1 = identical coverage)."""
    nv = np.isfinite(new.values)
    rv = np.isfinite(ref.values)
    inter = np.logical_and(nv, rv).sum()
    union = np.logical_or(nv, rv).sum()
    return float(inter / union) if union else 1.0


def compare(new_path, ref_path, new_var, ref_var, plots=None):
    new, nv = _open_var(new_path, new_var)
    ref, rv = _open_var(ref_path, ref_var)
    new, ref = align(new, ref)

    result = {"new": f"{os.path.basename(new_path)}:{nv}",
              "ref": f"{os.path.basename(ref_path)}:{rv}",
              "shape_new": dict(new.sizes), "mask_jaccard": mask_jaccard(new, ref)}

    if "time" in new.dims:
        # overall (all times pooled) + per-time
        result["overall"] = metrics(new.values.ravel(), ref.values.ravel())
        per_t = []
        for i in range(new.sizes["time"]):
            per_t.append(metrics(new.isel(time=i).values.ravel(),
                                 ref.isel(time=i).values.ravel()))
        result["per_time"] = per_t
        # regional monthly means
        result["new_monthly_mean"] = [float(np.nanmean(new.isel(time=i).values))
                                      for i in range(new.sizes["time"])]
        result["ref_monthly_mean"] = [float(np.nanmean(ref.isel(time=i).values))
                                      for i in range(new.sizes["time"])]
    else:
        result["overall"] = metrics(new.values.ravel(), ref.values.ravel())

    if plots:
        _make_plots(new, ref, nv, plots, result)
    return result


def _make_plots(new, ref, label, outdir, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    n2 = new.isel(time=0) if "time" in new.dims else new
    r2 = ref.isel(time=0) if "time" in ref.dims else ref
    d = (n2 - r2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    n2.plot(ax=ax[0]); ax[0].set_title(f"new {label}")
    r2.plot(ax=ax[1]); ax[1].set_title(f"ref {label}")
    d.plot(ax=ax[2], cmap="RdBu_r"); ax[2].set_title("new - ref")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{label}_maps.png"), dpi=110)
    plt.close(fig)

    a = new.values.ravel(); b = ref.values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].hexbin(b, a, gridsize=60, mincnt=1, cmap="viridis")
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
        ax[0].plot([lo, hi], [lo, hi], "r-", lw=1)
        ax[0].set_xlabel("reference"); ax[0].set_ylabel("new")
        ax[0].set_title(f"{label} 1:1")
        ax[1].hist(a - b, bins=80)
        ax[1].set_title(f"residuals (new-ref)  bias={np.mean(a-b):.3g}")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{label}_scatter_resid.png"), dpi=110)
        plt.close(fig)

    if "new_monthly_mean" in result:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["new_monthly_mean"], "o-", label="new")
        ax.plot(result["ref_monthly_mean"], "s--", label="ref")
        ax.set_xlabel("month index"); ax.set_ylabel(f"regional mean {label}")
        ax.legend(); ax.set_title("regional monthly mean")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{label}_timeseries.png"), dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new"); ap.add_argument("ref")
    ap.add_argument("--new-var", default=None)
    ap.add_argument("--ref-var", default=None)
    ap.add_argument("--plots", default=None, help="output dir for diagnostic plots")
    ap.add_argument("--json", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    res = compare(args.new, args.ref, args.new_var, args.ref_var, args.plots)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.json:
        with open(args.json, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
