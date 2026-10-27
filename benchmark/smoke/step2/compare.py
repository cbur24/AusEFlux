# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "netCDF4"]
# ///
"""Bit-compare every *.nc in dir A against dir B.

usage: compare.py A B [--years 2005-2006] [--recursive] [--rows 10:25]
With --years, files in B must equal the (year-)slice of the files in A.
With --rows S:E, files in B must equal latitude rows S:E of the files in A,
carry the global attribute auseflux_lat_rows="S:E", and B's climatology files
must be named {name}_r{S}-{E}.nc (compared with A's {name}.nc).
Exit code 1 on any mismatch.
"""
import argparse
import os
import sys

import netCDF4 as nc4
import numpy as np
import pandas as pd


def load(path):
    with nc4.Dataset(path) as d:
        d.set_auto_mask(False)
        names = [v for v in d.variables if v not in ('time', 'latitude', 'longitude', 'month')]
        out = {}
        for v in names:
            out[v] = np.asarray(d[v][:])
        t = None
        if 'time' in d.variables:
            t = pd.to_datetime(nc4.num2date(d['time'][:], d['time'].units,
                                            only_use_cftime_datetimes=False,
                                            only_use_python_datetimes=True))
        lat = np.asarray(d['latitude'][:]); lon = np.asarray(d['longitude'][:])
    return out, t, lat, lon


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a'); p.add_argument('b')
    p.add_argument('--years')
    p.add_argument('--recursive', action='store_true')
    p.add_argument('--rows')
    a = p.parse_args()
    rows = tuple(map(int, a.rows.split(':'))) if a.rows else None
    suffix = f'_r{rows[0]}-{rows[1]}' if rows else ''

    def b_name(f):
        if rows and os.path.dirname(f) == 'climatology':
            return f[:-3] + suffix + '.nc'
        return f

    def listing(root):
        res = []
        for dp, dn, fn in os.walk(root):
            for f in fn:
                if f.endswith('.nc'):
                    res.append(os.path.relpath(os.path.join(dp, f), root))
            if not a.recursive:
                break
        return sorted(res)

    fa, fb0 = listing(a.a), listing(a.b)
    inv = {b_name(f): f for f in fa}
    fb = sorted(inv.get(f, f + ' (B name)') for f in fb0)
    bad = 0
    if fa != fb:
        print('FILE SETS DIFFER: only A', sorted(set(fa) - set(fb)), 'only B', sorted(set(fb) - set(fa)))
        bad += 1
    for f in sorted(set(fa) & set(fb)):
        va, ta, la, lo = load(os.path.join(a.a, f))
        pb = os.path.join(a.b, b_name(f))
        vb, tb, lb, lob = load(pb)
        msg = []
        with nc4.Dataset(pb) as d:
            lr = getattr(d, 'auseflux_lat_rows', None)
        if lr != (a.rows if rows else None):
            msg.append(f'auseflux_lat_rows={lr!r}')
        if rows:
            la = la[rows[0]:rows[1]]
            va = {k: (v[..., rows[0]:rows[1], :] if v.ndim >= 2 else v)
                  for k, v in va.items()}
        if not (np.array_equal(la, lb) and np.array_equal(lo, lob)):
            msg.append('coords')
        if a.years and ta is not None:
            y0, y1 = map(int, a.years.split('-'))
            sel = (ta.year >= y0) & (ta.year <= y1)
            ta = ta[sel]
            va = {k: v[sel] for k, v in va.items()}
        if (ta is None) != (tb is None) or (ta is not None and not ta.equals(tb)):
            msg.append(f'time {None if ta is None else (ta[0], ta[-1], len(ta))} vs '
                       f'{None if tb is None else (tb[0], tb[-1], len(tb))}')
        if set(va) != set(vb):
            msg.append(f'vars {sorted(va)} vs {sorted(vb)}')
        for k in set(va) & set(vb):
            x, y = va[k], vb[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                msg.append(f'{k} shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}')
                continue
            if x.dtype.kind == 'f':
                eq = x.view(f'u{x.dtype.itemsize}') == y.view(f'u{y.dtype.itemsize}')
                # treat any NaN payload as equal
                eq |= np.isnan(x) & np.isnan(y)
            else:
                eq = x == y
            if not eq.all():
                d = np.nanmax(np.abs(x.astype('f8') - y.astype('f8')))
                msg.append(f'{k}: {np.size(eq) - eq.sum()} differing, max|d|={d:.3g}')
        if msg:
            bad += 1
            print(f'DIFF {f}: ' + '; '.join(msg))
    print(f'compared {len(set(fa) & set(fb))} files: {"OK, bit-identical" if not bad else f"{bad} problems"}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
