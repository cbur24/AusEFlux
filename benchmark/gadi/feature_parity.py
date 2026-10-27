# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas"]
# ///
"""
Feature-stage / harmonisation-stage parity (experiments E3 and E4).

Default (features, step2 outputs): for each `{var}_500m.nc` present in both
NEW_DIR and REF_DIR (28 reference stacks, incl. the unicode `ΔT_500m.nc`),
align on (year, month) restricted to --years, and on coordinates (1e-4 deg),
stream over latitude chunks and report per variable and per month:
  R², RMSE, mean bias, max|d| over jointly finite pixels; NaN-mask mismatch
  count (finite in one, not in the other); and in each file separately the
  fraction of NaN and of +-Inf pixels.

--interim (step1 outputs): compares `{dir}/{var}/{var}_500m_{year}.nc` for
every var directory present in both and every year in --years (time steps
keyed by (year, month), or (year, month, day) for sub-monthly files), and adds
an orientation diagnostic per variable: Pearson correlation of the
1/10-resolution time-mean field new vs ref, and new vs vertically flipped ref
(a flipped-orientation bug shows corr_flipped >> corr).

Every common data variable in a file pair is compared (for single-variable
files with different names, those two are paired).  Output: markdown (stdout
and --out_md) and JSON (--json).

    uv run feature_parity.py NEW/features_500m REF/data/500m --years 2024-2025 \
        --out_md e3.md --json e3.json
    uv run feature_parity.py NEW/interim_500m REF/data/interim_500m --interim \
        --years 2024-2025 --vars LST,rain
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (CoarseMean, Field, GridMap, GroupStats, align_time,  # noqa: E402
                     choose_rows, data_vars, fmt, key_str, log, md_table,
                     open_ds, parse_int_spec, write_json)


def var_pairs(new_path, ref_path):
    dn = open_ds(new_path)
    dr = open_ds(ref_path)
    vn, vr = data_vars(dn), data_vars(dr)
    dn.close()
    dr.close()
    common = [v for v in vn if v in vr]
    if common:
        return [(v, v) for v in common]
    if len(vn) == 1 and len(vr) == 1:
        return [(vn[0], vr[0])]
    return []


def compare_pair(label, new_paths, ref_paths, vnew, vref, years, rows, orient):
    nf = Field(new_paths, vnew)
    rf = Field(ref_paths, vref)
    grid = GridMap(nf, rf)
    keys, only_n, only_r = align_time(
        nf, rf, (lambda k: k == ('static',) or k[0] in years) if years else None)
    T = len(keys)
    st = GroupStats(T, hist=False)
    cnt = {k: np.zeros(T, dtype=np.int64) for k in
           ('total', 'new_nan', 'new_inf', 'ref_nan', 'ref_inf', 'new_only', 'ref_only')}
    cm_n = CoarseMean(grid.y.n, grid.x.n, 10) if orient else None
    cm_r = CoarseMean(grid.y.n, grid.x.n, 10) if orient else None
    xn, xr_ = grid.xsel()
    for a, b, (ysn, ysr) in grid.row_chunks(choose_rows(rows, [nf, rf], 10)):
        A = nf.read(keys, ysn, xn)
        B = rf.read(keys, ysr, xr_)
        fa, fb = np.isfinite(A), np.isfinite(B)
        both = fa & fb
        sh = (1, 2)
        cnt['total'] += A.shape[1] * A.shape[2]
        cnt['new_nan'] += np.isnan(A).sum(sh)
        cnt['new_inf'] += np.isinf(A).sum(sh)
        cnt['ref_nan'] += np.isnan(B).sum(sh)
        cnt['ref_inf'] += np.isinf(B).sum(sh)
        cnt['new_only'] += (fa & ~fb).sum(sh)
        cnt['ref_only'] += (~fa & fb).sum(sh)
        na, nb, nboth = fa.sum(sh), fb.sum(sh), both.sum(sh)
        st.add_mask_groups(na, nb, nboth, na + nb - nboth)
        st.add(A[both], B[both], np.repeat(np.arange(T), nboth))
        if orient:
            y0 = a - grid.y.n0
            cm_n.add(y0, np.where(fa, A, 0).sum(0, dtype='float64'), fa.sum(0))
            cm_r.add(y0, np.where(fb, B, 0).sum(0, dtype='float64'), fb.sum(0))
    per_month = []
    for t, k in enumerate(keys):
        r = st.result(t)
        tot = cnt['total'][t]
        per_month.append(dict(
            month=key_str(k), n_both=r['n_pixels'], r2=r.get('r2'), rmse=r.get('rmse'),
            mean_bias=r.get('mean_bias'), max_abs_diff=r.get('max_abs_diff'),
            mask_mismatch=int(cnt['new_only'][t] + cnt['ref_only'][t]),
            new_only_finite=int(cnt['new_only'][t]), ref_only_finite=int(cnt['ref_only'][t]),
            new_nan_frac=cnt['new_nan'][t] / tot, new_inf_frac=cnt['new_inf'][t] / tot,
            ref_nan_frac=cnt['ref_nan'][t] / tot, ref_inf_frac=cnt['ref_inf'][t] / tot))
    # pooled over months: merge group sums into one
    pooled = GroupStats(1, hist=False)
    for s in GroupStats.SUMS:
        getattr(pooled, s)[0] = getattr(st, s).sum()
    pooled.maxad[0] = st.maxad.max() if T else -np.inf
    pooled.bmin[0] = st.bmin.min() if T else np.inf
    pooled.bmax[0] = st.bmax.max() if T else -np.inf
    pooled.add_mask(st.na.sum(), st.nb.sum(), st.nboth.sum(), st.nunion.sum())
    pr = pooled.result(0)
    tot = int(cnt['total'].sum())
    summary = dict(
        name=label, var_new=vnew, var_ref=vref, months=len(keys),
        only_new_months=[key_str(k) for k in only_n],
        only_ref_months=[key_str(k) for k in only_r],
        n_both=pr['n_pixels'], r2=pr.get('r2'), rmse=pr.get('rmse'),
        mean_bias=pr.get('mean_bias'), max_abs_diff=pr.get('max_abs_diff'),
        ref_mean=pr.get('ref_mean'), ref_std=pr.get('ref_std'),
        mask_jaccard=pr.get('mask_jaccard'),
        mask_mismatch=int(cnt['new_only'].sum() + cnt['ref_only'].sum()),
        new_nan_frac=cnt['new_nan'].sum() / tot if tot else None,
        new_inf_frac=cnt['new_inf'].sum() / tot if tot else None,
        ref_nan_frac=cnt['ref_nan'].sum() / tot if tot else None,
        ref_inf_frac=cnt['ref_inf'].sum() / tot if tot else None,
        grid=grid.to_dict())
    if orient:
        mn, mr = cm_n.mean(), cm_r.mean()
        summary['corr'] = _corr(mn, mr)
        summary['corr_ref_flipped'] = _corr(mn, mr[::-1, :])
        summary['orientation'] = (
            'FLIPPED?' if summary['corr_ref_flipped'] is not None and summary['corr'] is not None
            and summary['corr_ref_flipped'] > summary['corr'] + 0.1 else 'ok')
    nf.close()
    rf.close()
    return summary, per_month


def _pct(x):
    return None if x is None else 100.0 * float(x)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    x, y = a[m], b[m]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('new_dir')
    ap.add_argument('ref_dir')
    ap.add_argument('--years', default=None, help='year range/list, e.g. 2024-2025')
    ap.add_argument('--vars', default=None,
                    help='comma-separated variable subset (file stem before _500m, '
                         'or interim directory name)')
    ap.add_argument('--interim', action='store_true',
                    help='compare step1 layout {dir}/{var}/{var}_500m_{year}.nc and '
                         'add the orientation diagnostic')
    ap.add_argument('--rows', type=int, default=400, help='latitude rows per chunk')
    ap.add_argument('--out_md', default=None, help='write markdown here')
    ap.add_argument('--json', default=None, help='write JSON here')
    args = ap.parse_args()

    t0 = time.time()
    years = parse_int_spec(args.years) if args.years else None
    want = set(v.strip() for v in args.vars.split(',')) if args.vars else None
    jobs = []      # (label, new_paths, ref_paths)
    missing = {'only_new': [], 'only_ref': []}
    if not args.interim:
        stem = lambda p: os.path.basename(p)[:-len('_500m.nc')]  # noqa: E731
        nv = {stem(p): p for p in glob.glob(os.path.join(args.new_dir, '*_500m.nc'))}
        rv = {stem(p): p for p in glob.glob(os.path.join(args.ref_dir, '*_500m.nc'))}
        missing['only_new'] = sorted(set(nv) - set(rv))
        missing['only_ref'] = sorted(set(rv) - set(nv))
        for v in sorted(set(nv) & set(rv)):
            if want is None or v in want:
                jobs.append((v, [nv[v]], [rv[v]]))
    else:
        dirs = lambda d: {os.path.basename(p) for p in glob.glob(os.path.join(d, '*'))  # noqa: E731
                          if os.path.isdir(p)}
        dn, dr = dirs(args.new_dir), dirs(args.ref_dir)
        missing['only_new'] = sorted(dn - dr)
        missing['only_ref'] = sorted(dr - dn)
        for v in sorted(dn & dr):
            if want is not None and v not in want:
                continue
            yrs = years
            if yrs is None:
                pick = lambda d: {int(os.path.basename(p)[len(v) + 6:-3])  # noqa: E731
                                  for p in glob.glob(os.path.join(d, v, f'{v}_500m_*.nc'))
                                  if os.path.basename(p)[len(v) + 6:-3].isdigit()}
                yrs = sorted(pick(args.new_dir) & pick(args.ref_dir))
            np_ = [os.path.join(args.new_dir, v, f'{v}_500m_{y}.nc') for y in yrs]
            rp_ = [os.path.join(args.ref_dir, v, f'{v}_500m_{y}.nc') for y in yrs]
            ok = [i for i in range(len(yrs))
                  if os.path.exists(np_[i]) and os.path.exists(rp_[i])]
            if not ok:
                log(f'  {v}: no common years, skipped')
                continue
            jobs.append((v, [np_[i] for i in ok], [rp_[i] for i in ok]))
    if not jobs:
        sys.exit('ERROR: nothing to compare')

    results = []
    for label, npaths, rpaths in jobs:
        pairs = var_pairs(npaths[0], rpaths[0])
        if not pairs:
            log(f'  {label}: no comparable data variables, skipped')
            continue
        for vn, vr in pairs:
            name = label if len(pairs) == 1 else f'{label}:{vn}'
            try:
                s, pm = compare_pair(name, npaths, rpaths, vn, vr, years, args.rows,
                                     args.interim)
            except (ValueError, KeyError) as e:
                log(f'  {name}: ERROR {e}')
                results.append(dict(summary=dict(name=name, error=str(e)), per_month=[]))
                continue
            results.append(dict(summary=s, per_month=pm))
            log(f'  {name}: R2={fmt(s["r2"], 6)} max|d|={fmt(s["max_abs_diff"])} '
                f'mask mismatch={s["mask_mismatch"]} ({time.time() - t0:.0f}s)')

    title = 'interim (step1)' if args.interim else 'features (step2)'
    L = [f'# feature_parity: {title}', '',
         f'- new: `{os.path.abspath(args.new_dir)}`',
         f'- ref: `{os.path.abspath(args.ref_dir)}`',
         f'- years: {args.years or "all common"}',
         f'- only in new: {missing["only_new"] or "none"}; only in ref: '
         f'{missing["only_ref"] or "none"}', '', '## Summary (pooled over months)', '']
    hdr = ['variable', 'months', 'n jointly finite', 'R²', 'RMSE', 'bias', 'max abs Δ',
           'NaN-mask mismatch', '% NaN new', '% Inf new', '% NaN ref', '% Inf ref']
    if args.interim:
        hdr += ['corr', 'corr (ref flipped)', 'orientation']
    rows = []
    for r in results:
        s = r['summary']
        if 'error' in s:
            rows.append([s['name'], 'ERROR: ' + s['error']] + [''] * (len(hdr) - 2))
            continue
        row = [s['name'], s['months'], s['n_both'], s['r2'], s['rmse'], s['mean_bias'],
               s['max_abs_diff'], s['mask_mismatch']] + [
                   _pct(s[k]) for k in ('new_nan_frac', 'new_inf_frac', 'ref_nan_frac',
                                         'ref_inf_frac')]
        if args.interim:
            row += [s.get('corr'), s.get('corr_ref_flipped'), s.get('orientation')]
        rows.append(row)
    L += [md_table(hdr, rows), '', '## Per month', '']
    for r in results:
        if not r['per_month']:
            continue
        L += [f'### {r["summary"]["name"]}', '',
              md_table(['month', 'n', 'R²', 'RMSE', 'bias', 'max abs Δ', 'mask mismatch',
                        '% NaN new', '% Inf new', '% NaN ref', '% Inf ref'],
                       [[m['month'], m['n_both'], m['r2'], m['rmse'], m['mean_bias'],
                         m['max_abs_diff'], m['mask_mismatch']] + [
                           _pct(m[k]) for k in ('new_nan_frac', 'new_inf_frac',
                                                'ref_nan_frac', 'ref_inf_frac')]
                        for m in r['per_month']]), '']
    md = '\n'.join(L) + '\n'
    print(md)
    if args.out_md:
        with open(args.out_md, 'w') as f:
            f.write(md)
    if args.json:
        write_json(args.json, dict(mode=title, new=args.new_dir, ref=args.ref_dir,
                                   years=years, missing=missing, results=results,
                                   wall_s=time.time() - t0))
    log(f'done ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
