"""
Download bbox subsets of all AusEFlux raw inputs (and the published v2.1
product) from NCI THREDDS via OPeNDAP, into the directory layout that
step1_harmonisation.py expects.

OPeNDAP transfers only the requested spatial/temporal slice, so a continental
~5 GB yearly file becomes a few MB for a small bbox. Files are written with
CF attributes preserved (mask_and_scale=False) so the harmonisation step
decodes them exactly as it would the full continental file.

Usage (run inside the local uv env):
    uv run fetch_inputs.py --years 2003-2022 --target-year 2020
    uv run fetch_inputs.py --years 2019-2020 --target-year 2020 --skip-product
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["xarray", "netCDF4", "pydap", "numpy"]
# ///

import argparse
import os
import sys
import time

import numpy as np
import xarray as xr

THREDDS = "https://thredds.nci.org.au/thredds/dodsC/ub8/au"

# Local roots mirroring the constants in step1/step3 of the pipeline.
INPUT_ROOT = "/home/prl900/Downloads/wald_eflux/input"
# Published reference product (Level-4 ground truth) kept separate.
REF_ROOT = "/home/prl900/Downloads/wald_eflux/reference/AusEFlux_v2.1"

# Default benchmark bbox: SE Australia (Tumbarumba / Cumberland Plain).
DEF_W, DEF_E, DEF_S, DEF_N = 148.0, 151.0, -36.0, -33.0

FLUXES = ["GPP", "ER", "NEE", "ET"]


def yearly_sources(year):
    """Return the THREDDS-relative paths of every raw input for one year.

    These paths mirror exactly the input layout step1_harmonisation.input_files()
    expects (with the 'ub8/au' prefix dropped), so files land where the pipeline
    reads them.
    """
    y = str(year)
    paths = [
        f"MODIS/mosaic/MCD43A4.006/MCD43A4.006.b01.500m_0620_0670nm_nbar.{y}.nc",
        f"MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.{y}.nc",
        f"MODIS/mosaic/MCD43A4.006/MCD43A4.006.b05.500m_1230_1250nm_nbar.{y}.nc",
        f"MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.{y}.nc",
        f"OzWALD/8day/NDVI/OzWALD.NDVI.{y}.nc",
        f"OzWALD/8day/LAI/OzWALD.LAI.{y}.nc",
        f"OzWALD/daily/meteo/Tmin/OzWALD.Tmin.{y}.nc",
        f"OzWALD/daily/meteo/Tmax/OzWALD.Tmax.{y}.nc",
        f"OzWALD/daily/meteo/kTavg/OzWALD.kTavg.{y}.nc",
        f"SILO/radiation/{y}.radiation.nc",
        f"SILO/daily_rain/{y}.daily_rain.nc",
        f"SILO/vp/{y}.vp.nc",
    ]
    if year < 2022:  # MYD11A1 QC layer only used/needed before 2022
        paths.append(f"MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.{y}.nc")
    return paths


def _coord_names(ds):
    lat = next((c for c in ("latitude", "lat", "y") if c in ds.coords or c in ds.dims), None)
    lon = next((c for c in ("longitude", "lon", "x") if c in ds.coords or c in ds.dims), None)
    if lat is None or lon is None:
        raise ValueError(f"could not find lat/lon in {list(ds.dims)}")
    return lat, lon


def _bbox_subset(ds, w, e, s, n):
    lat, lon = _coord_names(ds)
    desc = bool(ds[lat][0] > ds[lat][-1])
    lat_slice = slice(n, s) if desc else slice(s, n)
    return ds.sel({lat: lat_slice, lon: slice(w, e)})


def fetch(rel_src, local_path, w, e, s, n, tsel=None, overwrite=False):
    if os.path.exists(local_path) and not overwrite:
        print(f"  exists, skip: {local_path}")
        return "skip"
    url = f"{THREDDS}/{rel_src}"
    t0 = time.time()
    try:
        # decode_cf=False keeps raw values + CF attrs so step1 decodes identically.
        ds = xr.open_dataset(url, decode_cf=False)
    except Exception as ex:  # noqa: BLE401
        print(f"  ERROR opening {url}: {ex}")
        return "error"
    try:
        sub = _bbox_subset(ds, w, e, s, n)
        if tsel is not None and "time" in sub.dims:
            sub = sub.sel(time=tsel)
        sub = sub.load()
    except Exception as ex:  # noqa: BLE401
        print(f"  ERROR subsetting {url}: {ex}")
        ds.close()
        return "error"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    sub.to_netcdf(local_path)
    sz = os.path.getsize(local_path) / 1e6
    ny = sub.sizes.get(_coord_names(sub)[0], "?")
    nx = sub.sizes.get(_coord_names(sub)[1], "?")
    print(f"  ok ({time.time()-t0:4.1f}s, {sz:6.1f} MB, {ny}x{nx}): {local_path}")
    ds.close()
    return "ok"


def parse_years(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2003-2022",
                    help="harmonisation period, e.g. 2003-2022 or 2019,2020")
    ap.add_argument("--target-year", type=int, default=2020,
                    help="year of the published product to fetch for Level 4")
    ap.add_argument("--west", type=float, default=DEF_W)
    ap.add_argument("--east", type=float, default=DEF_E)
    ap.add_argument("--south", type=float, default=DEF_S)
    ap.add_argument("--north", type=float, default=DEF_N)
    ap.add_argument("--skip-product", action="store_true",
                    help="do not download the published v2.1 product")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    years = parse_years(args.years)
    w, e, s, n = args.west, args.east, args.south, args.north
    print(f"bbox W{w} E{e} S{s} N{n}  years {years[0]}-{years[-1]}  target {args.target_year}")

    tally = {"ok": 0, "skip": 0, "error": 0}
    for year in years:
        print(f"\n=== {year} ===")
        for rel in yearly_sources(year):
            r = fetch(rel, os.path.join(INPUT_ROOT, rel),
                      w, e, s, n, overwrite=args.overwrite)
            tally[r] = tally.get(r, 0) + 1

    if not args.skip_product:
        print(f"\n=== published v2.1 product ({args.target_year}) ===")
        for flux in FLUXES:
            rel = (f"AusEFlux/v2.1/monthly/{flux}/"
                   f"AusEFlux_{flux}_500m_quantiles_{args.target_year}_v2.1.nc")
            local = os.path.join(REF_ROOT,
                                 f"AusEFlux_{flux}_500m_quantiles_{args.target_year}_v2.1.nc")
            r = fetch(rel, local, w, e, s, n, overwrite=args.overwrite)
            tally[r] = tally.get(r, 0) + 1

    print(f"\nDONE: {tally}")
    if tally["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
