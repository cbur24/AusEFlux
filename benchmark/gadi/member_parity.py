# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas"]
# ///
"""
Per-member parity between two sets of ensemble-member predictions
(experiment E2a: new step3 --dump_members vs the reference member files).

For every member name present in both NEW_DIR and REF_DIR (e.g.
GPP_lgbm_1, GPP_rf_15) the files are aligned on (year, month) and on
coordinates (1e-4 deg) and streamed over latitude chunks.  A member may be a
single `{name}.nc` or month parts `{name}_m{MM}.nc` (also under `parts/`, as
written by step3 --months --dump_members DIR -> DIR/{flux}/parts/).

Per member and month: max|d| and mean|d| over jointly finite pixels, number
of jointly valid pixels, NaN-mask mismatches (new-only / ref-only valid);
per member: the 10 largest differences (lat, lon, month, new, ref).
Writes a markdown summary (stdout and --out_md) and JSON (--json).

    uv run member_parity.py NEW/GPP REF/2025/GPP --flux GPP --months 1-3 \
        --out_md e2a_GPP.md --json e2a_GPP.json
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (Field, GridMap, TopK, align_time, choose_rows,  # noqa: E402
                     find_member_files, fmt, key_str, log, md_table,
                     member_names, parse_months, write_json)


def _natural(name):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def compare_member(name, new_paths, ref_paths, var, months, rows, topk, tol):
    nf = Field(new_paths, var)
    rf = Field(ref_paths, var)
    grid = GridMap(nf, rf, tol=tol)
    keys, only_n, only_r = align_time(
        nf, rf, (lambda k: k[1] in months) if months else None)
    T = len(keys)
    mx = np.zeros(T)
    sabs = np.zeros(T)
    nboth = np.zeros(T, dtype=np.int64)
    new_only = np.zeros(T, dtype=np.int64)
    ref_only = np.zeros(T, dtype=np.int64)
    n_equal = np.zeros(T, dtype=np.int64)
    top = TopK(topk)
    xn, xr_ = grid.xsel()
    for a, b, (ysn, ysr) in grid.row_chunks(choose_rows(rows, [nf, rf])):
        for t, k in enumerate(keys):
            A = nf.read([k], ysn, xn)
            B = rf.read([k], ysr, xr_)
            va, vb = np.isfinite(A), np.isfinite(B)
            both = va & vb
            new_only[t] += int((va & ~vb).sum())
            ref_only[t] += int((~va & vb).sum())
            nboth[t] += int(both.sum())
            if both.any():
                d = np.abs(A[both].astype('float64') - B[both].astype('float64'))
                mx[t] = max(mx[t], float(d.max()))
                sabs[t] += float(d.sum())
                n_equal[t] += int((d == 0).sum())
                top.add(np.where(both, A, np.nan), np.where(both, B, np.nan),
                        [k], grid.lat[a - grid.y.n0:b - grid.y.n0], grid.lon)
    per_month = [dict(month=key_str(k), n_both=int(nboth[t]), n_equal=int(n_equal[t]),
                      max_abs_diff=float(mx[t]),
                      mean_abs_diff=float(sabs[t] / nboth[t]) if nboth[t] else None,
                      mask_mismatch=int(new_only[t] + ref_only[t]),
                      new_only_valid=int(new_only[t]), ref_only_valid=int(ref_only[t]))
                 for t, k in enumerate(keys)]
    nb = int(nboth.sum())
    summary = dict(member=name, months=len(keys), n_both=nb,
                   n_equal=int(n_equal.sum()),
                   max_abs_diff=float(mx.max()) if T else None,
                   mean_abs_diff=float(sabs.sum() / nb) if nb else None,
                   mask_mismatch=int(new_only.sum() + ref_only.sum()),
                   only_new_months=[key_str(k) for k in only_n],
                   only_ref_months=[key_str(k) for k in only_r],
                   grid=grid.to_dict())
    tops = [dict(abs_diff=i[0], lat=i[1], lon=i[2], month=i[3], new=i[4], ref=i[5])
            for i in top.items]
    nf.close()
    rf.close()
    return summary, per_month, tops


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('new_dir')
    ap.add_argument('ref_dir')
    ap.add_argument('--flux', required=True,
                    help='member prefix and variable name (e.g. GPP)')
    ap.add_argument('--months', default=None, help='month subset, e.g. "1-3"')
    ap.add_argument('--members', default=None,
                    help='comma-separated member names to restrict to')
    ap.add_argument('--tol', type=float, default=1e-5,
                    help='max|d| tolerance for the PASS column (default 1e-5)')
    ap.add_argument('--top', type=int, default=10, help='largest differences listed')
    ap.add_argument('--rows', type=int, default=400, help='latitude rows per chunk')
    ap.add_argument('--out_md', default=None, help='write markdown here')
    ap.add_argument('--json', default=None, help='write JSON here')
    args = ap.parse_args()

    t0 = time.time()
    months = parse_months(args.months)
    names = sorted(member_names(args.new_dir, args.flux)
                   & member_names(args.ref_dir, args.flux), key=_natural)
    if args.members:
        want = set(args.members.split(','))
        names = [n for n in names if n in want]
    only_new = sorted(member_names(args.new_dir, args.flux)
                      - member_names(args.ref_dir, args.flux))
    only_ref = sorted(member_names(args.ref_dir, args.flux)
                      - member_names(args.new_dir, args.flux))
    if not names:
        sys.exit('ERROR: no common member names')
    log(f'{len(names)} common members; only new: {only_new}; only ref: {only_ref}')

    results = []
    for nm in names:
        s, pm, tops = compare_member(nm, find_member_files(args.new_dir, nm),
                                     find_member_files(args.ref_dir, nm),
                                     args.flux, months, args.rows, args.top, 1e-4)
        s['pass'] = (s['max_abs_diff'] is not None and s['max_abs_diff'] <= args.tol
                     and s['mask_mismatch'] == 0)
        results.append(dict(summary=s, per_month=pm, top=tops))
        log(f'  {nm}: max|d|={fmt(s["max_abs_diff"], 6)} mean|d|={fmt(s["mean_abs_diff"], 6)} '
            f'mask mismatch={s["mask_mismatch"]} ({time.time() - t0:.0f}s)')

    L = [f'# member_parity: {args.flux}', '',
         f'- new: `{os.path.abspath(args.new_dir)}`',
         f'- ref: `{os.path.abspath(args.ref_dir)}`',
         f'- members compared: {len(names)}; only in new: {only_new or "none"}; '
         f'only in ref: {only_ref or "none"}',
         f'- PASS = max abs Δ <= {args.tol:g} and identical NaN masks', '',
         '## Summary', '',
         md_table(['member', 'months', 'n jointly valid', '% bit-equal', 'max abs Δ',
                   'mean abs Δ', 'NaN-mask mismatch', 'result'],
                  [[r['summary']['member'], r['summary']['months'],
                    r['summary']['n_both'],
                    100.0 * r['summary']['n_equal'] / r['summary']['n_both']
                    if r['summary']['n_both'] else None,
                    r['summary']['max_abs_diff'], r['summary']['mean_abs_diff'],
                    r['summary']['mask_mismatch'], r['summary']['pass']]
                   for r in results]), '']
    L += ['## Per member and month', '']
    for r in results:
        L += [f'### {r["summary"]["member"]}', '',
              md_table(['month', 'n jointly valid', 'max abs Δ', 'mean abs Δ',
                        'NaN-mask mismatch', 'new-only', 'ref-only'],
                       [[m['month'], m['n_both'], m['max_abs_diff'], m['mean_abs_diff'],
                         m['mask_mismatch'], m['new_only_valid'], m['ref_only_valid']]
                        for m in r['per_month']]), '']
        if r['top']:
            L += [f'Largest {len(r["top"])} differences:', '',
                  md_table(['abs Δ', 'lat', 'lon', 'month', 'new', 'ref'],
                           [[t['abs_diff'], f'{t["lat"]:.4f}', f'{t["lon"]:.4f}',
                             t['month'], t['new'], t['ref']] for t in r['top']]), '']
    md = '\n'.join(L) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
    if args.json:
        write_json(args.json, dict(flux=args.flux, new=args.new_dir, ref=args.ref_dir,
                                   members=results, only_new=only_new, only_ref=only_ref,
                                   wall_s=time.time() - t0))
    ok = all(r['summary']['pass'] for r in results)
    log(f'ALL PASS: {"YES" if ok else "NO"} ({time.time() - t0:.1f}s)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
