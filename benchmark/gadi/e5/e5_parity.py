"""Per-month parity of E5 step2 features vs the reference features on a window.
usage: e5_parity.py WINDOW YEAR NEWDIR PART NPARTS OUTJSON
NEWDIR is either the continental dir (rows selected) or a windowed dir (auseflux_lat_rows)."""
import glob, json, os, sys, time
import numpy as np
import netCDF4 as nc4
import pandas as pd

REF = '/g/data/xc0/project/AusEFlux/data/500m'
WIN = {'north': (600, 1000), 'centre': (2800, 3200), 'south': (5000, 5400)}
win, year, newdir, part, nparts, outj = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
a0, a1 = WIN[win]


def load(path, rows):
    ds = nc4.Dataset(path)
    lr = getattr(ds, 'auseflux_lat_rows', None)
    if lr is not None:
        assert lr == f'{a0}:{a1}', (path, lr)
        rows = None
    v = [k for k, x in ds.variables.items() if x.ndim >= 2][0]
    var = ds.variables[v]
    var.set_auto_mask(False)
    tv = ds.variables['time']
    t = pd.DatetimeIndex(nc4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                                      only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    sel = np.where(t.year == year)[0]
    keys = [(int(y), int(m)) for y, m in zip(t.year[sel], t.month[sel])]
    sl = slice(sel[0], sel[-1] + 1)
    r = slice(*rows) if rows else slice(None)
    arr = np.asarray(var[sl, r, :], dtype='float32')
    lat = np.asarray(ds.variables['latitude'][r])
    ds.close()
    return arr, keys, lat


def monthly(A, B):
    ax = (1, 2)
    fa, fb = np.isfinite(A), np.isfinite(B)
    both = fa & fb
    d = np.where(both, A.astype('f8') - B.astype('f8'), 0.0)
    b = np.where(both, B.astype('f8'), 0.0)
    ad = np.abs(d)
    out = dict(n=both.sum(ax), mask_mismatch=(fa != fb).sum(ax),
               new_only=(fa & ~fb).sum(ax), ref_only=(~fa & fb).sum(ax),
               nfpat_diff=((np.isnan(A) != np.isnan(B)) | (np.isposinf(A) != np.isposinf(B)) |
                           (np.isneginf(A) != np.isneginf(B))).sum(ax),
               n_equal=((A == B) & both).sum(ax),
               n_rel1e3=((ad > 1e-3 * np.maximum(np.abs(b), 1e-6)) & both).sum(ax),
               sse=(d * d).sum(ax), sd=d.sum(ax), sb=b.sum(ax), sb2=(b * b).sum(ax),
               max_abs=ad.max(ax))
    return {k: v.tolist() for k, v in out.items()}


t0 = time.time()
names = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{newdir}/*_500m.nc'))
names = names[part::nparts]
out = {}
for name in names:
    rp = f'{REF}/{name}_500m.nc'
    if not os.path.exists(rp):
        print('no ref for', name, flush=True); continue
    B, kb, latb = load(rp, (a0, a1))
    A, ka, lata = load(f'{newdir}/{name}_500m.nc', (a0, a1))
    assert np.abs(lata - latb).max() < 1e-4, (name, 'lat')
    assert ka == kb, (name, ka[:2], kb[:2])
    m = monthly(A, B)
    out[name] = dict(keys=ka, **m)
    print(f'{win} {year} {name}: n={sum(m["n"])} eq={sum(m["n_equal"])} >1e-3rel={sum(m["n_rel1e3"])} '
          f'max|d|={max(m["max_abs"]):.3g} mm={sum(m["mask_mismatch"])} ({time.time()-t0:.0f}s)', flush=True)
    json.dump(out, open(outj, 'w'))
print('done', time.time() - t0)
