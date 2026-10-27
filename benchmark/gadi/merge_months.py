# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas"]
# ///
"""
Assemble month part files written by a sharded step3 run
(`step3_ensemble_prediction.py --months M`) into the final annual product.

Looks for `AusEFlux_{flux}_500m_quantiles_{year}_{version}_m{MM}.nc` in
PARTS_DIR (or PARTS_DIR/{flux}/parts, PARTS_DIR/parts), checks that every
expected month (default 1-12) is present exactly once with the right year,
that all parts share the grid (bit-identical coordinates) and variables, then
streams the data (latitude chunks) into OUT with the product layout
(float32, NaN fill, zlib, int time as days since 15 January of the year, `spatial_ref` CRS variable; global and variable
attributes copied from the first part).  The written file is re-read and
verified bit-for-bit (including NaN masks) against the parts; exit status 1 on
any problem.

    uv run merge_months.py /scratch/.../results/AusEFlux/GPP/parts --flux GPP \
        --year 2024 --out /scratch/.../AusEFlux_GPP_500m_quantiles_2024_v2.1.nc
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (Field, create_nc, data_vars, key_str, log,  # noqa: E402
                     open_ds, parse_months)


def find_parts(parts_dir, flux, year, version):
    pat = f'AusEFlux_{flux}_500m_quantiles_{year}_{version}_m[0-9][0-9].nc'
    for d in (parts_dir, os.path.join(parts_dir, flux, 'parts'),
              os.path.join(parts_dir, 'parts')):
        files = sorted(glob.glob(os.path.join(d, pat)))
        if files:
            return files
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('parts_dir')
    ap.add_argument('--flux', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--out', required=True, help='final product path')
    ap.add_argument('--months', default='1-12', help='expected months (default 1-12)')
    ap.add_argument('--version', default='v2.1')
    ap.add_argument('--rows', type=int, default=400, help='latitude rows per chunk')
    ap.add_argument('--no_verify', action='store_true', help='skip the read-back check')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing OUT')
    args = ap.parse_args()

    t0 = time.time()
    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f'ERROR: {args.out} exists (use --overwrite)')
    files = find_parts(args.parts_dir, args.flux, args.year, args.version)
    if not files:
        sys.exit(f'ERROR: no part files for {args.flux} {args.year} under {args.parts_dir}')
    want = parse_months(args.months)
    log(f'{len(files)} part files')

    # --- structural checks -------------------------------------------------
    import netCDF4
    problems = []
    vnames = None
    lat = lon = None
    times = {}
    for p in files:
        ds = open_ds(p)
        dv = data_vars(ds)
        if vnames is None:
            vnames = dv
            lat, lon = ds['latitude'].values, ds['longitude'].values
        else:
            if sorted(dv) != sorted(vnames):
                problems.append(f'{os.path.basename(p)}: variables {dv} != {vnames}')
            if (not np.array_equal(ds['latitude'].values, lat)
                    or not np.array_equal(ds['longitude'].values, lon)):
                problems.append(f'{os.path.basename(p)}: coordinates differ')
        for t in ds['time'].values:
            ts = np.datetime64(t, 'D').astype(object)
            k = (ts.year, ts.month)
            if k in times:
                problems.append(f'month {key_str(k)} in {os.path.basename(p)} and '
                                f'{os.path.basename(times[k][1])}')
            times[k] = (t, p)
        ds.close()
    got = sorted(times)
    wrong_year = [k for k in got if k[0] != args.year]
    missing = [m for m in want if (args.year, m) not in times]
    extra = [k for k in got if k[0] == args.year and k[1] not in want]
    if wrong_year:
        problems.append(f'time steps outside {args.year}: {[key_str(k) for k in wrong_year]}')
    if missing:
        problems.append(f'missing months: {missing}')
    if extra:
        problems.append(f'unexpected months: {[key_str(k) for k in extra]}')
    if problems:
        for pr in problems:
            log(f'ERROR: {pr}')
        sys.exit(1)
    keys = [(args.year, m) for m in want]

    # --- write ---------------------------------------------------------------
    with netCDF4.Dataset(files[0]) as src:
        gattrs = {a: src.getncattr(a) for a in src.ncattrs()}
        vattrs = {v: {a: src[v].getncattr(a) for a in src[v].ncattrs()
                      if a not in ('_FillValue', 'missing_value', 'coordinates',
                                   'grid_mapping')}
                  for v in vnames}
    gattrs['history'] = (str(gattrs.get('history', '')) +
                         f'\nmerge_months.py: {len(files)} parts').strip()
    fields = {v: Field(files, v) for v in vnames}
    ny, nx = len(lat), len(lon)
    tmp = args.out + '.tmp'
    nc = create_nc(tmp, vnames, lat, lon, [times[k][0] for k in keys], gattrs, vattrs)
    try:
        for r in range(0, ny, args.rows):
            h = min(args.rows, ny - r)
            for v in vnames:
                nc[v][:, r:r + h, :] = fields[v].read(keys, (r, h, 1), (0, nx, 1))
    finally:
        nc.close()
    log(f'written {tmp} ({time.time() - t0:.1f}s)')

    # --- verify ---------------------------------------------------------------
    if not args.no_verify:
        bad = 0
        for v in vnames:
            out = Field(tmp, v)
            if out.keys != keys:
                log(f'ERROR: {v}: written months {out.keys} != {keys}')
                bad += 1
            for r in range(0, ny, args.rows):
                h = min(args.rows, ny - r)
                A = out.read(keys, (r, h, 1), (0, nx, 1))
                B = fields[v].read(keys, (r, h, 1), (0, nx, 1))
                # bitwise equal, NaN == NaN
                n = int((A.view('uint32') != B.view('uint32')).sum()
                        - (np.isnan(A) & np.isnan(B) &
                           (A.view('uint32') != B.view('uint32'))).sum())
                bad += n
            out.close()
        if bad:
            log(f'ERROR: read-back verification found {bad} differing values; '
                f'{tmp} left in place')
            sys.exit(1)
        log('read-back verification: identical to parts (incl. NaN masks)')
    os.replace(tmp, args.out)
    log(f'wrote {args.out}: {len(vnames)} variables x {len(keys)} months '
        f'({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
