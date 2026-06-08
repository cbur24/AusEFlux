"""
Extract the 2025 slice of the reference's continental 500 m feature datasets
from gadi, READ-ONLY (no writes on gadi), into local netCDF files that the
pipeline's step3 can consume.

Mechanism: a small python (gadi's xc0 miniconda, which has xarray+netCDF4) reads
the 2025 time-slice of each feature in lat-bands and streams raw float32 bytes
to stdout; we reshape and wrap them in a netCDF locally. Keeps gadi-side RAM low
(~200 MB) and writes nothing on gadi.

    uv run python extract_continental.py
"""
import os
import subprocess
import sys

import argparse
import numpy as np
import pandas as pd
import xarray as xr

GADI = "gadi"
GADI_PY = "/g/data/xc0/software/python/miniconda3/bin/python"
SRC = "/g/data/xc0/project/AusEFlux/data/500m"
OUT_ROOT = "/home/prl900/Downloads/wald_eflux"
LAT, LON, NT = 6800, 8400, 12

VARS_TXT = "/home/prl900/Downloads/wald_eflux/results/variables.txt"

# remote streamer: argv[1] = feature var, argv[2] = year; streams raw float32
# one full (lat, lon) grid per timestep -> time-major C-order matching
# reshape(NT, LAT, LON) locally. (Earlier a lat-band stream put all timesteps
# inside each band, which did NOT match the reshape and scrambled the data.)
REMOTE = r'''
import xarray as xr, sys
v, year = sys.argv[1], sys.argv[2]
d = xr.open_dataset("%s/%%s_500m.nc" %% v).sel(time=year)
dv = [x for x in d.data_vars if x not in ("crs", "spatial_ref")][0]
da = d[dv]
for t in range(da.sizes["time"]):
    sys.stdout.buffer.write(
        da.isel(time=t).values.astype("float32").tobytes())
''' % SRC

REMOTE_COORDS = r'''
import xarray as xr, sys
d = xr.open_dataset("%s/kNDVI_500m.nc")
sys.stdout.buffer.write(d.latitude.values.astype("float64").tobytes())
sys.stdout.buffer.write(d.longitude.values.astype("float64").tobytes())
''' % SRC


def need_vars():
    h = open(VARS_TXT).read().strip().split(",")
    return [v[:-3] for v in h if v and v != "site"]


def ssh_stream(remote_script, outfile, arg=""):
    with open(outfile, "wb") as f:
        p = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=40", GADI, f"{GADI_PY} - {arg}"],
            input=remote_script.encode(), stdout=f, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode()[-500:])
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", default="2025")
    ap.add_argument("--out", default=None,
                    help="output dir (default features_500m_cont for 2025, "
                         "features_500m_cont_<year> otherwise)")
    args = ap.parse_args()
    year = str(args.year)
    out = args.out or (f"{OUT_ROOT}/features_500m_cont" if year == "2025"
                       else f"{OUT_ROOT}/features_500m_cont_{year}")
    times = pd.to_datetime([f"{year}-{m:02d}-15" for m in range(1, 13)])
    os.makedirs(out, exist_ok=True)

    # 1. coords (once; same canonical grid for all years)
    cpath = os.path.join(out, "_coords.raw")
    if not os.path.exists(cpath) or os.path.getsize(cpath) != (LAT + LON) * 8:
        print("fetching coords...")
        assert ssh_stream(REMOTE_COORDS, cpath) == 0, "coord fetch failed"
    buf = np.fromfile(cpath, dtype="float64")
    lat, lon = buf[:LAT], buf[LAT:LAT + LON]

    for v in need_vars():
        out_nc = os.path.join(out, f"{v}_500m.nc")
        if os.path.exists(out_nc):
            print(f"  {v}: exists, skip")
            continue
        raw = os.path.join(out, f"{v}.raw")
        print(f"  {v}: streaming {year}...", flush=True)
        rc = ssh_stream(REMOTE, raw, arg=f"{v} {year}")
        exp = NT * LAT * LON * 4
        if rc != 0 or os.path.getsize(raw) != exp:
            print(f"    FAIL {v}: rc={rc} size={os.path.getsize(raw)} exp={exp}")
            if os.path.exists(raw):
                os.remove(raw)
            continue
        arr = np.fromfile(raw, dtype="float32").reshape(NT, LAT, LON)
        ds = xr.Dataset(
            {v: (("time", "latitude", "longitude"), arr)},
            coords={"time": times, "latitude": lat, "longitude": lon})
        ds.to_netcdf(out_nc)
        os.remove(raw)
        print(f"    wrote {out_nc} ({os.path.getsize(out_nc)/1e6:.0f} MB)")
    if os.path.exists(cpath):
        os.remove(cpath)
    print(f"extraction complete ({year}) -> {out}")


if __name__ == "__main__":
    main()
