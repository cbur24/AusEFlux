# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "netCDF4"]
# ///
"""
check_format.py NEW REF [--allow-extra-attrs a,b] [--strict-storage] [--any-year]

Structural comparison of a new AusEFlux product file against a published one:
dimensions, variable names and dtypes, coordinate values (bit-equal),
decoded time stamps, per-variable attributes (`_FillValue`, `grid_mapping`),
the `spatial_ref` CRS variable and its attributes, and global attributes.
Storage differences (contiguous vs chunked/deflated) are reported but only
fail with --strict-storage.  Exit 0 when the layout matches, 1 otherwise.
No data values are compared (use compare_products.py for that).
With --any-year the new file may hold a different year than REF (annual
update checked against last year's published file): the time axis must then
be the 15th of each of the 12 months of one year with units
"days since {year}-01-15", and the description/units are compared as usual.
"""
import argparse
import sys

import netCDF4 as nc4
import numpy as np


def _attrs(v):
    return {a: v.getncattr(a) for a in v.ncattrs()}


def _eq(a, b):
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (np.isnan(a) and np.isnan(b)) or a == b
    try:
        return bool(np.array_equal(a, b)) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else a == b
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('new')
    ap.add_argument('ref')
    ap.add_argument('--allow-extra-attrs', default='history,auseflux_quantile_method,'
                    'auseflux_mask_vars,auseflux_mask_test,auseflux_n_models,'
                    'auseflux_valid_only,auseflux_lat_rows,auseflux_partial_months',
                    help='global attributes the new file may carry in addition')
    ap.add_argument('--strict-storage', action='store_true',
                    help='fail on chunking/compression differences')
    ap.add_argument('--any-year', action='store_true',
                    help='allow NEW to hold a different year than REF (same '
                         'mid-month pattern and units template)')
    args = ap.parse_args()
    extra_ok = set(a for a in args.allow_extra_attrs.split(',') if a)
    problems, notes = [], []

    with nc4.Dataset(args.new) as n, nc4.Dataset(args.ref) as r:
        n.set_auto_mask(False); r.set_auto_mask(False)
        # dimensions
        nd = {k: len(v) for k, v in n.dimensions.items()}
        rd = {k: len(v) for k, v in r.dimensions.items()}
        if nd != rd:
            problems.append(f'dimensions {nd} != {rd}')
        # variables
        nv, rv = set(n.variables), set(r.variables)
        if nv != rv:
            problems.append(f'variables differ: missing {sorted(rv - nv)}, extra {sorted(nv - rv)}')
        for name in sorted(nv & rv):
            a, b = n[name], r[name]
            if a.dtype != b.dtype:
                problems.append(f'{name}: dtype {a.dtype} != {b.dtype}')
            if a.dimensions != b.dimensions:
                problems.append(f'{name}: dims {a.dimensions} != {b.dimensions}')
            aa, ba = _attrs(a), _attrs(b)
            for k in sorted(set(aa) | set(ba)):
                if k not in aa:
                    problems.append(f'{name}: attribute {k!r} missing (ref {ba[k]!r})')
                elif k not in ba:
                    problems.append(f'{name}: extra attribute {k!r} = {aa[k]!r}')
                elif not _eq(aa[k], ba[k]):
                    problems.append(f'{name}: attribute {k!r} {aa[k]!r} != {ba[k]!r}')
            if name in ('latitude', 'longitude'):
                if not np.array_equal(a[:], b[:]):
                    d = np.abs(a[:].astype('f8') - b[:].astype('f8')).max()
                    problems.append(f'{name}: values differ (max |d| {d:g})')
            elif name == 'time':
                ta = nc4.num2date(a[:], a.units, getattr(a, 'calendar', 'standard'))
                tb = nc4.num2date(b[:], b.units, getattr(b, 'calendar', 'standard'))
                if args.any_year:
                    years = sorted({t.year for t in ta})
                    want = [f'{years[0]}-{m:02d}-15 00:00:00' for m in range(1, 13)] \
                        if len(years) == 1 else []
                    if [str(t) for t in ta] != want:
                        problems.append(f'time: expected the 12 mid-month stamps of one '
                                        f'year, got {[str(t) for t in ta]}')
                    if a.units != f'days since {years[0]}-01-15':
                        problems.append(f'time: units {a.units!r} != '
                                        f"'days since {years[0]}-01-15'")
                else:
                    if [str(t) for t in ta] != [str(t) for t in tb]:
                        problems.append(f'time: decoded stamps differ: {ta} vs {tb}')
                    if a.units != b.units:
                        problems.append(f'time: units {a.units!r} != {b.units!r}')
            elif a.ndim == 0:
                pass
            else:
                sa = (a.chunking(), a.filters())
                sb = (b.chunking(), b.filters())
                if sa != sb:
                    msg = f'{name}: storage new {sa} vs ref {sb}'
                    (problems if args.strict_storage else notes).append(msg)
        # global attributes
        ga, gb = _attrs(n), _attrs(r)
        for k in sorted(set(ga) | set(gb)):
            if k not in ga:
                problems.append(f'global attribute {k!r} missing (ref {gb[k]!r})')
            elif k not in gb:
                (notes if k in extra_ok else problems).append(f'extra global attribute {k!r} = {str(ga[k])[:80]!r}')
            elif not _eq(ga[k], gb[k]):
                problems.append(f'global attribute {k!r}: {str(ga[k])[:120]!r} != {str(gb[k])[:120]!r}')
        if n.data_model != r.data_model:
            problems.append(f'format {n.data_model} != {r.data_model}')

    for m in notes:
        print(f'NOTE  {m}')
    for m in problems:
        print(f'FAIL  {m}')
    print('FORMAT: ' + ('PASS' if not problems else f'FAIL ({len(problems)} problems)'))
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
