"""
Continental comparison: local 2025 output vs the published v2.1 product on the
ub8/THREDDS OPeNDAP server, streamed per time-step to bound memory (each
6800x8400 slice ~0.23 GB). Both are on the identical canonical 500 m grid, so
no regridding is needed (coords are checked to 4 d.p.).

    uv run python compare_continental.py --new-root <dir> --fluxes GPP NEE --year 2025
"""
import argparse
import numpy as np
import xarray as xr

THREDDS = ("https://thredds.nci.org.au/thredds/dodsC/ub8/au/AusEFlux/v2.1/"
           "monthly/{flux}/AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc")
BANDS = ["25th_percentile", "median", "75th_percentile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--fluxes", nargs="+", default=["GPP", "NEE"])
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()

    print("flux band            | R2      RMSE    MAE     bias    max|d|   n")
    for flux in args.fluxes:
        new = xr.open_dataset(
            f"{args.new_root}/{flux}/AusEFlux_{flux}_500m_quantiles_{args.year}_v2.1.nc")
        ref = xr.open_dataset(THREDDS.format(flux=flux, year=args.year))
        # align coords (identical canonical grid; sort both ascending for safety)
        for d in ("latitude", "longitude"):
            if bool(new[d][0] > new[d][-1]):
                new = new.sortby(d)
            if bool(ref[d][0] > ref[d][-1]):
                ref = ref.sortby(d)
        ref = ref.reindex(latitude=new.latitude, longitude=new.longitude,
                          method="nearest", tolerance=1e-3)
        for band in BANDS:
            v = f"{flux}_{band}"
            # streaming accumulators incl bias
            n_t = new[v].sizes["time"]
            ss_res = sad = s_b = s_b2 = s_d = 0.0
            smax = 0.0
            n = 0
            for t in range(n_t):
                a = new[v].isel(time=t).values.astype("float64").ravel()
                b = ref[v].isel(time=t).values.astype("float64").ravel()
                m = np.isfinite(a) & np.isfinite(b)
                a, b = a[m], b[m]
                d = a - b
                ss_res += float(np.sum(d * d)); sad += float(np.sum(np.abs(d)))
                s_d += float(np.sum(d)); s_b += float(np.sum(b))
                s_b2 += float(np.sum(b * b))
                if d.size:
                    smax = max(smax, float(np.abs(d).max()))
                n += a.size
            if n == 0:
                print(f"{flux:3s} {band:16s} | no overlapping valid pixels "
                      f"(check that {v} output is complete)")
                continue
            mean_b = s_b / n
            ss_tot = s_b2 - n * mean_b * mean_b
            r2 = 1 - ss_res / ss_tot
            print(f"{flux:3s} {band:16s} | {r2:.4f}  {np.sqrt(ss_res/n):6.3f}  "
                  f"{sad/n:6.3f}  {s_d/n:+.3f}  {smax:6.2f}  {n}")


if __name__ == "__main__":
    main()
