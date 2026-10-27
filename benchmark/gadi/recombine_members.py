# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas"]
# ///
"""
Recombine per-member ensemble predictions into the 25/50/75 percentile
product (experiment E1: isolate the quantile method).

Reads `{MEMBER_DIR}/{flux}_{lgbm,rf}_{i}.nc` (variable named after the flux,
dims time/latitude/longitude or time/y/x), streams over latitude chunks and
months, stacks the members and computes the quantiles with

  nearest  transcription of the reference `master:src/_percentile.py`
           np_percentile(nodata=NaN): sort along the member axis (NaN last),
           take index round_half_even(q * (n_valid - 1)); NaN where fewer than
           3 members are valid.  Matches step3 --quantile_method nearest.
  linear   np.nanquantile (step3 default), NaN where < 3 valid.

Default: write the product (`{flux}_25th_percentile`, `{flux}_median`,
`{flux}_75th_percentile`; dims time/latitude/longitude; float32, NaN fill,
zlib; time = 15th of month; coordinates copied from the members).

--check REF_PRODUCT: do not write; stream the recombination against REF
(aligned on (year, month) and coordinates within 1e-4 deg) and report per
band max|d| over jointly valid pixels, the NaN-mask mismatch count and the
number of exactly equal values.  Exit status 0 only if every band is
bit-identical (max|d| == 0 and no mask mismatch) -- E1's exactness gate.

Memory: n_members x tblock x rows x n_lon x 4 B (30 x 2 x 400 x 8400 x 4 B
~ 0.8 GB) plus a sorted copy.

    uv run recombine_members.py MEMBERS/ --flux GPP --method nearest --out GPP_2025_nearest.nc
    uv run recombine_members.py MEMBERS/ --flux GPP --method nearest --check REF.nc
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (BANDS, QUANTILES, Field, GridMap, align_time,  # noqa: E402
                     choose_rows, create_nc, fmt, key_str, log, md_table,
                     mid_month, parse_months, product_attrs, write_json)


def nearest_rank(stack, qs=QUANTILES):
    """stack (n, ...) float32 -> (len(qs), ...) like master:src/_percentile.py."""
    shp = stack.shape[1:]
    flat = stack.reshape(stack.shape[0], -1)
    vc = (~np.isnan(flat)).sum(axis=0)
    out = np.full((len(qs), flat.shape[1]), np.nan, dtype=stack.dtype)
    cols = np.flatnonzero(vc >= 3)
    if cols.size:
        srt = np.sort(flat[:, cols], axis=0)
        v = vc[cols]
        ar = np.arange(cols.size)
        for k, q in enumerate(qs):
            idx = np.round(q * (v - 1)).astype(np.int64)
            out[k, cols] = srt[idx, ar]
    return out.reshape((len(qs),) + shp)


def linear(stack, qs=QUANTILES):
    """np.nanquantile(stack, qs, axis=0) (linear), NaN where < 3 valid.

    numpy's nanquantile with an axis falls back to apply_along_axis (one
    Python call per pixel: hours per continental month).  Columns where all
    members are valid go through the vectorised np.quantile instead (the
    same per-column arithmetic, _quantile_unchecked); only partially valid
    columns use np.nanquantile.  `linear_selfcheck` verifies bit-equality
    against np.nanquantile on a random column sample of every chunk.
    """
    n = stack.shape[0]
    shp = stack.shape[1:]
    flat = stack.reshape(n, -1)
    vc = np.isfinite(flat).sum(axis=0)
    out = np.full((len(qs), flat.shape[1]), np.nan, dtype='float32')
    full = np.flatnonzero(vc == n)
    part = np.flatnonzero((vc >= 3) & (vc < n))
    if full.size:
        out[:, full] = np.quantile(flat[:, full], list(qs), axis=0).astype('float32')
    if part.size:   # RuntimeWarnings are silenced globally in main()
        out[:, part] = np.nanquantile(flat[:, part], list(qs),
                                      axis=0).astype('float32')
    return out.reshape((len(qs),) + shp)


def linear_selfcheck(stack, q, rng, n=2000, qs=QUANTILES):
    """Assert `q` equals np.nanquantile on n random columns (bitwise)."""
    import warnings
    flat = stack.reshape(stack.shape[0], -1)
    qf = q.reshape(q.shape[0], -1)
    vc = np.isfinite(flat).sum(axis=0)
    cand = np.flatnonzero(vc >= 3)
    if cand.size == 0:
        return 0
    cols = rng.choice(cand, size=min(n, cand.size), replace=False)
    part = cand[vc[cand] < flat.shape[0]]
    if part.size:  # always include some partially valid columns
        cols = np.union1d(cols, part[:200])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        exp = np.nanquantile(flat[:, cols], list(qs), axis=0).astype('float32')
    got = qf[:, cols]
    if not np.array_equal(exp.view('uint32'), got.view('uint32')):
        sys.exit('ERROR: fast linear quantile differs from np.nanquantile')
    return cols.size


def parallel_cols(fun, stack, threads):
    """Apply a column-wise quantile function in `threads` column blocks
    (np.sort / np.partition release the GIL; result is identical)."""
    if threads <= 1:
        return fun(stack)
    from concurrent.futures import ThreadPoolExecutor
    n = stack.shape[0]
    shp = stack.shape[1:]
    flat = stack.reshape(n, -1)
    edges = np.linspace(0, flat.shape[1], threads + 1).astype(int)
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(lambda ab: fun(flat[:, ab[0]:ab[1]]),
                            zip(edges[:-1], edges[1:])))
    return np.concatenate(parts, axis=1).reshape((parts[0].shape[0],) + shp)


def member_files(member_dir, flux):
    pat = re.compile(rf'^{re.escape(flux)}_(lgbm|rf)_(\d+)\.nc$')
    files = sorted(p for p in glob.glob(os.path.join(member_dir, f'{flux}_*.nc'))
                   if pat.match(os.path.basename(p)))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('member_dir', help='directory with {flux}_{lgbm,rf}_{i}.nc')
    ap.add_argument('--flux', required=True)
    ap.add_argument('--method', choices=['nearest', 'linear'], default='nearest',
                    help='quantile method (default nearest = reference)')
    ap.add_argument('--out', default=None, help='output product file (write mode)')
    ap.add_argument('--check', default=None, metavar='REF_PRODUCT',
                    help='compare against this product instead of writing')
    ap.add_argument('--json', default=None, help='with --check: write results JSON here')
    ap.add_argument('--months', default=None, help='month subset, e.g. "1-12" or "1,6"')
    ap.add_argument('--year', type=int, default=None,
                    help='use only time steps of this year (e.g. 2024 from the '
                         '264-step historical member files; only those steps are read)')
    ap.add_argument('--threads', type=int,
                    default=int(os.environ.get('PBS_NCPUS', '1')),
                    help='threads for the quantile computation (column blocks; '
                         'default $PBS_NCPUS or 1)')
    ap.add_argument('--n_models', type=int, default=None,
                    help='use only the first N members (sorted by file name)')
    ap.add_argument('--rows', type=int, default=400, help='latitude rows per chunk')
    ap.add_argument('--tblock', type=int, default=2,
                    help='months read together per chunk (default 2)')
    args = ap.parse_args()
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    if (args.out is None) == (args.check is None):
        ap.error('give exactly one of --out or --check')

    t0 = time.time()
    files = member_files(args.member_dir, args.flux)
    if args.n_models:
        files = files[:args.n_models]
    if len(files) < 3:
        sys.exit(f'ERROR: found {len(files)} member files for {args.flux} in '
                 f'{args.member_dir} (need >= 3)')
    log(f'{args.flux}: {len(files)} members, method {args.method}')
    members = [Field(p, args.flux) if _has_var(p, args.flux)
               else Field(p) for p in files]
    m0 = members[0]
    for f, p in zip(members[1:], files[1:]):
        if (len(f.lat) != len(m0.lat) or len(f.lon) != len(m0.lon)
                or np.abs(f.lat - m0.lat).max() > 1e-4
                or np.abs(f.lon - m0.lon).max() > 1e-4):
            sys.exit(f'ERROR: grid of {p} differs from {files[0]}')
    months = parse_months(args.months)
    keys = [k for k in m0.keys if all(k in f.index for f in members)
            and (months is None or k[1] in months)
            and (args.year is None or k[0] == args.year)]
    if not keys:
        sys.exit('ERROR: no common months across members')
    qbase = nearest_rank if args.method == 'nearest' else linear
    rng = np.random.default_rng(0)
    n_selfcheck = 0

    def qfun(stack):
        nonlocal n_selfcheck
        q = parallel_cols(qbase, stack, args.threads)
        if args.method == 'linear':
            n_selfcheck += linear_selfcheck(stack, q, rng)
        return q
    log(f'months {", ".join(key_str(k) for k in keys)}; threads {args.threads}')
    vnames = [f'{args.flux}_{b}' for b in BANDS]
    ny, nx = len(m0.lat), len(m0.lon)

    if args.out:
        rows = choose_rows(args.rows, members)
        attrs = product_attrs(args.flux, extra={
            'history': f'recombine_members.py --method {args.method} '
                       f'({len(files)} members from {os.path.abspath(args.member_dir)})'})
        nc = create_nc(args.out + '.tmp', vnames, m0.lat_raw, m0.lon_raw,
                       [mid_month(*k[:2]) for k in keys], attrs)
        try:
            for r in range(0, ny, rows):
                h = min(rows, ny - r)
                for tb in range(0, len(keys), args.tblock):
                    kk = keys[tb:tb + args.tblock]
                    stack = np.stack([f.read(kk, (r, h, 1), (0, nx, 1)) for f in members])
                    q = qfun(stack)
                    for b, vn in enumerate(vnames):
                        nc[vn][tb:tb + len(kk), r:r + h, :] = q[b]
                log(f'  rows {r}-{r + h} ({time.time() - t0:.0f}s)')
        finally:
            nc.close()
        os.replace(args.out + '.tmp', args.out)
        if n_selfcheck:
            log(f'linear self-check vs np.nanquantile: {n_selfcheck} columns bit-equal')
        log(f'wrote {args.out} ({len(keys)} months, {time.time() - t0:.1f}s)')
        return

    # --- check mode ---
    refs = {b: Field(args.check, f'{args.flux}_{b}') for b in BANDS}
    grid = GridMap(m0, refs[BANDS[0]])
    keys, only_m, only_r = align_time(m0, refs[BANDS[0]],
                                      lambda k: k in keys)
    if only_r:
        log(f'WARNING: months only in REF: {[key_str(k) for k in only_r]}')
    rows = choose_rows(args.rows, members + list(refs.values()))
    st = {b: dict(n_both=0, n_equal=0, mask_mismatch=0, new_only=0, ref_only=0,
                  max_abs_diff=0.0) for b in BANDS}
    xs_n, xs_r = grid.xsel()
    for a, b_, (ysn, ysr) in grid.row_chunks(rows):
        for tb in range(0, len(keys), args.tblock):
            kk = keys[tb:tb + args.tblock]
            stack = np.stack([f.read(kk, ysn, xs_n) for f in members])
            q = qfun(stack)
            del stack
            for bi, band in enumerate(BANDS):
                R = refs[band].read(kk, ysr, xs_r)
                A = q[bi]
                va, vr = np.isfinite(A), np.isfinite(R)
                both = va & vr
                s = st[band]
                s['n_both'] += int(both.sum())
                s['new_only'] += int((va & ~vr).sum())
                s['ref_only'] += int((~va & vr).sum())
                s['mask_mismatch'] = s['new_only'] + s['ref_only']
                # bitwise equality on jointly valid values
                s['n_equal'] += int((A[both].view('uint32') == R[both].view('uint32')).sum())
                if both.any():
                    s['max_abs_diff'] = max(s['max_abs_diff'], float(
                        np.abs(A[both].astype('float64') - R[both].astype('float64')).max()))
        log(f'  rows {a}-{b_} ({time.time() - t0:.0f}s)')
    if n_selfcheck:
        log(f'linear self-check vs np.nanquantile: {n_selfcheck} columns bit-equal')
    exact = all(s['max_abs_diff'] == 0 and s['mask_mismatch'] == 0
                and s['n_equal'] == s['n_both'] for s in st.values())
    rows_md = [[b, s['n_both'], s['n_equal'], s['max_abs_diff'], s['mask_mismatch'],
                s['new_only'], s['ref_only'],
                s['max_abs_diff'] == 0 and s['mask_mismatch'] == 0] for b, s in st.items()]
    print()
    print(f'recombine ({args.method}, {len(files)} members) vs {args.check}; '
          f'months {", ".join(key_str(k) for k in keys)}')
    print(md_table(['band', 'n jointly valid', 'n bit-equal', 'max abs Δ', 'NaN-mask mismatch',
                    'new-only valid', 'ref-only valid', 'exact'], rows_md))
    print(f'\nEXACT: {"YES" if exact else "NO"}  ({time.time() - t0:.1f}s)')
    if args.json:
        write_json(args.json, dict(method=args.method, members=files, ref=args.check,
                                   months=[key_str(k) for k in keys], bands=st,
                                   exact=exact, wall_s=time.time() - t0))
    sys.exit(0 if exact else 1)


def _has_var(path, var):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return var in ds.variables


if __name__ == '__main__':
    main()
