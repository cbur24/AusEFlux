"""
Subset the continental, grid-aligned static rasters to the benchmark bbox so
step1's target grid (taken from the land/sea mask) becomes the small test
region instead of the whole continent.

The land/sea mask defines the canonical 500 m grid. We subset it to a buffered
bbox, then reindex ndvi_of_baresoil and urban_water_mask onto that exact
subgrid (nearest) so all three share identical coordinates. VegH (1 km) and the
C4 .tif are reprojected on the fly by the pipeline, so they are left untouched.

Originals are moved to static/_full/ and the bbox versions written in their place.

Usage:
    uv run python prep_static.py                      # default buffered bbox
    uv run python prep_static.py --west 147.9 --east 151.1 --south -36.1 --north -32.9
"""

import argparse
import os
import shutil

import xarray as xr

STATIC = "/home/prl900/Downloads/wald_eflux/input/static"
FULL = os.path.join(STATIC, "_full")

# Grid-aligned rasters that MUST match the target grid exactly.
GRID_ALIGNED = {
    "land_sea_mask_500m.nc": "landsea_mask_500m",
    "ndvi_of_baresoil_500m.nc": "NDVI",
    "urban_water_mask_500m.nc": "urban_water_mask",
}

# Buffered bbox: ~20 px (0.1 deg) beyond the 148-151 / -36..-33 product extent,
# so reprojection edge effects fall outside the comparison region.
DEF_W, DEF_E, DEF_S, DEF_N = 147.9, 151.1, -36.1, -32.9


def _latlon(ds):
    lat = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else "y")
    lon = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else "x")
    return lat, lon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--west", type=float, default=DEF_W)
    ap.add_argument("--east", type=float, default=DEF_E)
    ap.add_argument("--south", type=float, default=DEF_S)
    ap.add_argument("--north", type=float, default=DEF_N)
    args = ap.parse_args()

    os.makedirs(FULL, exist_ok=True)

    # Move originals aside (idempotent: only move if not already in _full).
    for fn in GRID_ALIGNED:
        src, dst = os.path.join(STATIC, fn), os.path.join(FULL, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
        elif os.path.exists(src) and os.path.exists(dst):
            os.remove(src)  # regenerating; drop stale bbox version

    # 1) Subset the land/sea mask -> defines the canonical target subgrid.
    mfn = "land_sea_mask_500m.nc"
    mds = xr.open_dataset(os.path.join(FULL, mfn))
    lat, lon = _latlon(mds)
    desc = bool(mds[lat][0] > mds[lat][-1])
    lat_slice = slice(args.north, args.south) if desc else slice(args.south, args.north)
    msub = mds.sel({lat: lat_slice, lon: slice(args.west, args.east)})
    msub.to_netcdf(os.path.join(STATIC, mfn))
    print(f"{mfn}: {dict(msub.sizes)}  "
          f"{lat} {float(msub[lat].min()):.4f}..{float(msub[lat].max()):.4f}  "
          f"{lon} {float(msub[lon].min()):.4f}..{float(msub[lon].max()):.4f}")
    target_lat = msub[lat]
    target_lon = msub[lon]
    mds.close()

    # 2) Reindex the other two onto the mask subgrid (nearest) -> identical coords.
    for fn in ("ndvi_of_baresoil_500m.nc", "urban_water_mask_500m.nc"):
        ds = xr.open_dataset(os.path.join(FULL, fn))
        la, lo = _latlon(ds)
        sub = ds.sel({la: lat_slice, lo: slice(args.west, args.east)})
        sub = sub.reindex({la: target_lat.values, lo: target_lon.values}, method="nearest")
        sub.to_netcdf(os.path.join(STATIC, fn))
        print(f"{fn}: {dict(sub.sizes)}")
        ds.close()

    print("done.")


if __name__ == "__main__":
    main()
