"""
Level 4 - end-to-end equivalence: compare the new pipeline's final flux product
against the published v2.1 product on NCI THREDDS, for every flux and band.

Writes a markdown report + per-band diagnostic plots.

    uv run python level4_report.py --year 2020
"""

import argparse
import os

import compare as C  # local module

NEW_ROOT = "/home/prl900/Downloads/wald_eflux/results/AusEFlux"
REF_ROOT = "/home/prl900/Downloads/wald_eflux/reference/AusEFlux_v2.1"
OUT = os.path.dirname(os.path.abspath(__file__))

FLUXES = ["GPP", "ER", "NEE", "ET"]
BANDS = ["25th_percentile", "median", "75th_percentile"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    y = args.year

    rows = []
    plotdir = os.path.join(OUT, "report_plots", str(y))
    for flux in FLUXES:
        new = os.path.join(NEW_ROOT, flux,
                           f"AusEFlux_{flux}_500m_quantiles_{y}_v2.1.nc")
        ref = os.path.join(REF_ROOT,
                           f"AusEFlux_{flux}_500m_quantiles_{y}_v2.1.nc")
        if not os.path.exists(new):
            print(f"  MISSING new product: {new}")
            continue
        if not os.path.exists(ref):
            print(f"  MISSING ref product: {ref}")
            continue
        for band in BANDS:
            var = f"{flux}_{band}"
            res = C.compare(new, ref, var, var,
                            plots=os.path.join(plotdir, var) if args.plots else None)
            ov = res.get("overall", {})
            rows.append((flux, band, ov.get("n_pixels"), ov.get("r2"),
                         ov.get("rmse"), ov.get("mae"), ov.get("mean_bias"),
                         ov.get("max_abs_diff"), ov.get("pct_within_1pct_range"),
                         res.get("mask_jaccard")))
            print(f"{flux:4s} {band:16s} R2={ov.get('r2')} rmse={ov.get('rmse')} "
                  f"bias={ov.get('mean_bias')} jac={res.get('mask_jaccard')}")

    # markdown
    md = [f"# Level 4 - new pipeline vs published v2.1 ({y})", "",
          "| flux | band | n_pix | R² | RMSE | MAE | mean bias | max|Δ| | % within 1% range | mask Jaccard |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flux, band, n, r2, rmse, mae, bias, mx, pct, jac = r
        def f(x, d=4):
            return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
        md.append(f"| {flux} | {band} | {n} | {f(r2,5)} | {f(rmse)} | {f(mae)} | "
                  f"{f(bias)} | {f(mx)} | {f(pct,2)} | {f(jac,4)} |")
    md.append("")
    path = os.path.join(OUT, f"LEVEL4_{y}.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
