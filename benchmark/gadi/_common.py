"""
Shared helpers for the benchmark/gadi tools (imported by the sibling scripts;
not run directly).

Everything here is built for streaming over the continental 500 m grid
(time, 6800, 8400) without ever holding a whole variable in memory:

  * `Field`      one variable, possibly spread over several files (e.g. month
                 part files), indexed by (year, month) keys; chunked reads.
  * `match_axis` / `GridMap`   coordinate alignment with a 1e-4 deg tolerance,
                 cropping to the overlap and handling flipped orientation.
  * `align_time` (year, month) key alignment (time encodings may differ).
  * `GroupStats` float64 streaming accumulators of agreement metrics (R2,
                 RMSE, MAE, bias, max|d|, % within 1 % of range, mask Jaccard)
                 for K groups at once (overall / per month / per MVG) via
                 np.bincount.  Metric definitions follow benchmark/compare.py.
  * `TopK`       the K largest |new - ref| differences with their location.
  * `create_nc`  netCDF4 writer matching the product layout of
                 step3_ensemble_prediction.py (float32, NaN fill, zlib).
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

LAT_NAMES = ('latitude', 'lat', 'y')
LON_NAMES = ('longitude', 'lon', 'x')
SKIP_VARS = {'crs', 'spatial_ref'}
BANDS = ['25th_percentile', 'median', '75th_percentile']
QUANTILES = (0.25, 0.5, 0.75)
COORD_TOL = 1e-4
EARTH_R_KM = 6371.0072

FLUX_METADATA = {
    'GPP': {'long_name': 'Gross Primary Productivity',
            'units': 'gC/m\N{SUPERSCRIPT TWO}/month'},
    'NEE': {'long_name': 'Net Ecosystem Exchange',
            'units': 'gC/m\N{SUPERSCRIPT TWO}/month'},
    'ER': {'long_name': 'Ecosystem Respiration',
           'units': 'gC/m\N{SUPERSCRIPT TWO}/month'},
    'ET': {'long_name': 'Evapotranspiration', 'units': 'mm/month'},
}


# ---------------------------------------------------------------------------
# small parsers / formatting
# ---------------------------------------------------------------------------

def parse_int_spec(spec, lo=None, hi=None):
    """'1-12', '1,6,12', '1-3,7' -> sorted list of ints."""
    out = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = (int(x) for x in part.split('-', 1))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f'empty specification {spec!r}')
    if lo is not None and min(out) < lo or hi is not None and max(out) > hi:
        raise ValueError(f'specification {spec!r} out of range [{lo}, {hi}]')
    return sorted(out)


def parse_months(spec):
    return None if spec is None else parse_int_spec(spec, 1, 12)


def fmt(x, d=4):
    if x is None:
        return 'n/a'
    if isinstance(x, (bool, np.bool_)):
        return 'PASS' if x else 'FAIL'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax != 0 and (ax < 10 ** -d or ax >= 1e7):
            return f'{float(x):.{d}g}' if d > 2 else f'{float(x):.3g}'
        return f'{float(x):.{d}f}'
    return str(x)


def md_table(headers, rows, digits=None):
    """Markdown table; R²/Jaccard/corr columns get 6 decimals by default."""
    dg = []
    for h in headers:
        d = 6 if any(t in h for t in ('R²', 'Jaccard', 'corr')) else 4
        dg.append((digits or {}).get(h, d))
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(fmt(v, d) if not isinstance(v, str) else v
                                     for v, d in zip(r, dg)) + ' |')
    return '\n'.join(out)


def jsonable(o):
    """Recursively convert numpy scalars / NaN for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# dataset / variable access
# ---------------------------------------------------------------------------

def open_ds(path):
    """Lazy open; cache=False so slices are never kept in memory."""
    try:
        return xr.open_dataset(path, cache=False)
    except (ValueError, TypeError):
        # undecodable time units: fall back to cftime
        return xr.open_dataset(path, cache=False, use_cftime=True)


def find_dims(da):
    tdim = 'time' if 'time' in da.dims else None
    ydim = next((d for d in LAT_NAMES if d in da.dims), None)
    xdim = next((d for d in LON_NAMES if d in da.dims), None)
    if ydim is None or xdim is None:
        raise ValueError(f'{da.name}: cannot find lat/lon dims in {da.dims}')
    return tdim, ydim, xdim


def data_vars(ds):
    return [v for v in ds.data_vars
            if v not in SKIP_VARS and ds[v].ndim >= 2]


def pick_var(ds, preferred=None):
    if preferred is not None and preferred in ds.data_vars:
        return preferred
    cands = data_vars(ds)
    if preferred is not None:
        raise KeyError(f'variable {preferred!r} not found (have {cands})')
    if not cands:
        raise KeyError('no 2-D/3-D data variable found')
    return cands[0]


def time_keys(ds, tdim, mode='ym'):
    if tdim is None:
        return [('static',)]
    t = ds[tdim]
    y = t.dt.year.values
    m = t.dt.month.values
    if mode == 'ym':
        return [(int(a), int(b)) for a, b in zip(y, m)]
    d = t.dt.day.values
    return [(int(a), int(b), int(c)) for a, b, c in zip(y, m, d)]


def key_str(k):
    if k == ('static',):
        return 'static'
    return '-'.join(f'{v:02d}' if i else str(v) for i, v in enumerate(k))


class Field:
    """
    One gridded variable, possibly spread over several files that share the
    grid (e.g. month part files).  Time steps are addressed by keys
    ((year, month) or (year, month, day), see `keymode`).  Reads return
    float32 arrays (time, rows, cols) in (time, lat, lon) order.
    """

    def __init__(self, paths, var=None, keymode='auto'):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.parts = []
        self.index = {}
        self.var = None
        for p in self.paths:
            ds = open_ds(p)
            v = pick_var(ds, var if var is not None else self.var)
            self.var = self.var or v
            da = ds[v]
            tdim, ydim, xdim = find_dims(da)
            da = da.transpose(*([tdim] if tdim else []), ydim, xdim)
            self.parts.append((ds, da, tdim, ydim, xdim))
        ds0, da0, tdim, ydim, xdim = self.parts[0]
        self.tdim = tdim
        self.lat = np.asarray(ds0[ydim].values, dtype='float64')
        self.lon = np.asarray(ds0[xdim].values, dtype='float64')
        self.lat_raw = ds0[ydim].values
        self.lon_raw = ds0[xdim].values
        for (ds, da, _, y, x), p in zip(self.parts[1:], self.paths[1:]):
            if (len(ds[y]) != len(self.lat) or len(ds[x]) != len(self.lon)
                    or np.abs(ds[y].values - self.lat).max() > COORD_TOL
                    or np.abs(ds[x].values - self.lon).max() > COORD_TOL):
                raise ValueError(f'{p}: grid differs from {self.paths[0]}')
        if keymode == 'auto':
            ym = [k for i in range(len(self.parts))
                  for k in time_keys(self.parts[i][0], self.parts[i][2], 'ym')]
            keymode = 'ym' if len(set(ym)) == len(ym) else 'ymd'
        self.keymode = keymode
        for i, (ds, da, tdim, _, _) in enumerate(self.parts):
            for j, k in enumerate(time_keys(ds, tdim, keymode)):
                if k in self.index:
                    raise ValueError(f'duplicate time key {k} in {self.paths}')
                self.index[k] = (i, j if tdim else None)
        self.keys = sorted(self.index)
        enc = da0.encoding.get('chunksizes')
        self.chunk_rows = int(enc[-2]) if enc else 1
        self.dtype = da0.dtype
        self.attrs = dict(da0.attrs)
        self.global_attrs = dict(ds0.attrs)

    def read(self, keys, ysel, xsel):
        """
        keys: list of time keys; ysel/xsel: (start, count, step) with step
        +-1 in this field's own index space.  Returns float32
        (len(keys), count_y, count_x).
        """
        ny, nx = ysel[1], xsel[1]
        out = np.empty((len(keys), ny, nx), dtype='float32')
        ys, yflip = _sel_slice(ysel)
        xs, xflip = _sel_slice(xsel)
        # group keys by part file, preserving output order
        by_part = {}
        for o, k in enumerate(keys):
            i, j = self.index[k]
            by_part.setdefault(i, []).append((o, j))
        for i, lst in by_part.items():
            ds, da, tdim, ydim, xdim = self.parts[i]
            if tdim is None:
                arr = da.isel({ydim: ys, xdim: xs}).values[None]
            else:
                tj = [j for _, j in lst]
                if tj == list(range(tj[0], tj[0] + len(tj))):
                    tsel = slice(tj[0], tj[0] + len(tj))
                else:
                    tsel = np.asarray(tj)
                arr = da.isel({tdim: tsel, ydim: ys, xdim: xs}).values
            if yflip:
                arr = arr[:, ::-1, :]
            if xflip:
                arr = arr[:, :, ::-1]
            for n, (o, _) in enumerate(lst):
                out[o] = arr[n] if tdim else arr[0]
        return out

    def close(self):
        for ds, *_ in self.parts:
            ds.close()


def _sel_slice(sel):
    start, count, step = sel
    if step == 1:
        return slice(start, start + count), False
    lo = start - (count - 1)
    return slice(lo, start + 1), True


def find_member_files(directory, name):
    """
    Files for member `name` in `directory`: `{name}.nc`, otherwise month
    part files `{name}_m??.nc` (also looked up under `parts/`).
    """
    p = os.path.join(directory, f'{name}.nc')
    if os.path.exists(p):
        return [p]
    for d in (directory, os.path.join(directory, 'parts')):
        parts = sorted(glob.glob(os.path.join(d, f'{name}_m[0-9][0-9].nc')))
        if parts:
            return parts
    return []


def member_names(directory, flux=None):
    names = set()
    for d in (directory, os.path.join(directory, 'parts')):
        for p in glob.glob(os.path.join(d, '*.nc')):
            b = os.path.basename(p)[:-3]
            b = re.sub(r'_m\d\d$', '', b)
            if flux is None or b.startswith(f'{flux}_'):
                names.add(b)
    return names


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

class AxisMap:
    """new index i in [n0, n1) <-> ref index r0 + step * (i - n0)."""

    def __init__(self, n0, n1, r0, step, max_offset):
        self.n0, self.n1, self.r0, self.step = n0, n1, r0, step
        self.max_offset = max_offset

    @property
    def n(self):
        return self.n1 - self.n0

    def shrink(self, k):
        if k <= 0:
            return self
        if 2 * k >= self.n:
            raise ValueError(f'border {k} leaves no overlap (n={self.n})')
        return AxisMap(self.n0 + k, self.n1 - k, self.r0 + self.step * k,
                       self.step, self.max_offset)

    def sel(self, a, b):
        """(new_sel, ref_sel) for new rows [a, b) (absolute new indices)."""
        return ((a, b - a, 1),
                (self.r0 + self.step * (a - self.n0), b - a, self.step))

    def to_dict(self):
        return dict(new_start=self.n0, new_stop=self.n1, ref_start=self.r0,
                    ref_step=self.step, n=self.n,
                    max_coord_offset=self.max_offset)


def match_axis(new, ref, tol=COORD_TOL, name='axis'):
    new = np.asarray(new, dtype='float64')
    ref = np.asarray(ref, dtype='float64')
    if len(new) > 1 and len(ref) > 1:
        dn = float(np.median(np.diff(new)))
        dr = float(np.median(np.diff(ref)))
        if abs(abs(dn) - abs(dr)) > 1e-3 * abs(dn):
            raise ValueError(f'{name}: grid spacing differs '
                             f'(new {dn:.8g}, ref {dr:.8g})')
    order = np.argsort(ref, kind='stable')
    rs = ref[order]
    pos = np.clip(np.searchsorted(rs, new), 1, len(rs) - 1) if len(rs) > 1 \
        else np.zeros(len(new), dtype=int)
    if len(rs) > 1:
        left = rs[pos - 1]
        right = rs[pos]
        pick = np.where(np.abs(new - left) <= np.abs(new - right), pos - 1, pos)
    else:
        pick = pos
    j = order[pick]
    off = np.abs(ref[j] - new)
    ok = off <= tol
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise ValueError(f'{name}: no overlapping coordinates within {tol}')
    if not np.all(np.diff(idx) == 1):
        raise ValueError(f'{name}: overlap is not contiguous')
    jj = j[idx]
    step = int(jj[1] - jj[0]) if len(jj) > 1 else 1
    if step not in (1, -1) or not np.all(np.diff(jj) == step):
        raise ValueError(f'{name}: matched indices not monotone (+-1)')
    return AxisMap(int(idx[0]), int(idx[-1]) + 1, int(jj[0]), step,
                   float(off[idx].max()))


class GridMap:
    def __init__(self, new_field, ref_field, border=0, tol=COORD_TOL):
        self.y = match_axis(new_field.lat, ref_field.lat, tol, 'latitude').shrink(border)
        self.x = match_axis(new_field.lon, ref_field.lon, tol, 'longitude').shrink(border)
        self.lat = new_field.lat[self.y.n0:self.y.n1]
        self.lon = new_field.lon[self.x.n0:self.x.n1]
        self.new_shape = (len(new_field.lat), len(new_field.lon))
        self.ref_shape = (len(ref_field.lat), len(ref_field.lon))

    def row_chunks(self, rows):
        for a in range(self.y.n0, self.y.n1, rows):
            b = min(a + rows, self.y.n1)
            yield a, b, self.y.sel(a, b)

    def xsel(self):
        return self.x.sel(self.x.n0, self.x.n1)

    def to_dict(self):
        return dict(new_shape=self.new_shape, ref_shape=self.ref_shape,
                    lat=self.y.to_dict(), lon=self.x.to_dict(),
                    lat_range=[float(self.lat.min()), float(self.lat.max())],
                    lon_range=[float(self.lon.min()), float(self.lon.max())])


def align_time(new_field, ref_field, select=None):
    """
    Common keys in new's order.  `select(key) -> bool` filters further.
    Returns (keys, only_new, only_ref).
    """
    if new_field.keymode != ref_field.keymode:
        raise ValueError('time key modes differ (monthly vs sub-monthly?)')
    common = [k for k in new_field.keys if k in ref_field.index]
    if select is not None:
        common = [k for k in common if select(k)]
    only_new = [k for k in new_field.keys if k not in ref_field.index]
    only_ref = [k for k in ref_field.keys if k not in new_field.index]
    return common, only_new, only_ref


def choose_rows(rows, fields, multiple=1):
    """Row-chunk size >= file chunk height (avoid re-decompressing)."""
    r = max([rows] + [f.chunk_rows for f in fields if f.chunk_rows < 10 ** 5])
    if multiple > 1:
        r = int(math.ceil(r / multiple) * multiple)
    return r


def cell_area_km2(lat, dlat, dlon, mode='sphere'):
    lat = np.asarray(lat, dtype='float64')
    c = np.cos(np.deg2rad(lat))
    if mode == '500m':
        return 0.25 * c
    return (EARTH_R_KM ** 2) * abs(np.deg2rad(dlat)) * abs(np.deg2rad(dlon)) * c


# ---------------------------------------------------------------------------
# streaming metric accumulators
# ---------------------------------------------------------------------------

HIST_LOGMIN, HIST_LOGMAX, HIST_NB = -12.0, 6.0, 36000  # 0.0005 decade bins
_HIST_W = (HIST_LOGMAX - HIST_LOGMIN) / HIST_NB


def _hist_bins(absd):
    """bin 0 = exact zero; 1..NB = log10 bins."""
    b = np.zeros(absd.shape, dtype=np.int64)
    nz = absd > 0
    if nz.any():
        lg = np.log10(absd[nz])
        b[nz] = 1 + np.clip(((lg - HIST_LOGMIN) / _HIST_W).astype(np.int64),
                            0, HIST_NB - 1)
    return b


def _count_below(hist, thr):
    """Approximate count of |d| < thr from the log histogram (exact zeros
    always counted; partial bin by log-linear interpolation)."""
    n = float(hist[0])
    if thr <= 0:
        return n
    lt = math.log10(thr)
    pos = (lt - HIST_LOGMIN) / _HIST_W
    if pos <= 0:
        return n
    full = int(min(math.floor(pos), HIST_NB))
    n += float(hist[1:1 + full].sum())
    if full < HIST_NB:
        n += float(hist[1 + full]) * (pos - full)
    return n


class GroupStats:
    """
    Streaming agreement statistics for K groups (float64 sums).
    new = a, ref = b, d = a - b over jointly finite values.
    Mask counts (for Jaccard) are added separately with `add_mask`.
    """
    SUMS = ('n', 'sa', 'sb', 'sa2', 'sb2', 'sab', 'sd', 'sd2', 'sad', 'sab_abs')

    def __init__(self, K=1, hist=True):
        self.K = K
        for s in self.SUMS:
            setattr(self, s, np.zeros(K))
        self.maxad = np.full(K, -np.inf)
        self.bmin = np.full(K, np.inf)
        self.bmax = np.full(K, -np.inf)
        self.hist = np.zeros((K, HIST_NB + 1)) if hist else None
        self.na = np.zeros(K)       # new valid
        self.nb = np.zeros(K)       # ref valid
        self.nboth = np.zeros(K)
        self.nunion = np.zeros(K)

    def add(self, a, b, labels=None):
        if a.size == 0:
            return
        a = a.astype('float64', copy=False)
        b = b.astype('float64', copy=False)
        d = a - b
        ad = np.abs(d)
        vals = {'n': None, 'sa': a, 'sb': b, 'sa2': a * a, 'sb2': b * b,
                'sab': a * b, 'sd': d, 'sd2': d * d, 'sad': ad,
                'sab_abs': np.abs(b)}
        if self.K == 1 or labels is None:
            for s, v in vals.items():
                getattr(self, s)[0] += a.size if v is None else float(v.sum())
            self.maxad[0] = max(self.maxad[0], float(ad.max()))
            self.bmin[0] = min(self.bmin[0], float(b.min()))
            self.bmax[0] = max(self.bmax[0], float(b.max()))
            if self.hist is not None:
                self.hist[0] += np.bincount(_hist_bins(ad), minlength=HIST_NB + 1)
            return
        labels = np.asarray(labels, dtype=np.int64)
        K = self.K
        for s, v in vals.items():
            getattr(self, s)[:] += np.bincount(labels, weights=v, minlength=K)[:K]
        df = pd.DataFrame({'g': labels, 'ad': ad, 'b': b})
        g = df.groupby('g', sort=False)
        mx = g['ad'].max()
        bmn = g['b'].min()
        bmx = g['b'].max()
        idx = mx.index.values
        self.maxad[idx] = np.maximum(self.maxad[idx], mx.values)
        self.bmin[idx] = np.minimum(self.bmin[idx], bmn.values)
        self.bmax[idx] = np.maximum(self.bmax[idx], bmx.values)
        if self.hist is not None:
            comb = labels * (HIST_NB + 1) + _hist_bins(ad)
            self.hist += np.bincount(comb, minlength=K * (HIST_NB + 1)) \
                .reshape(K, HIST_NB + 1)

    def add_mask(self, na, nb, nboth, nunion, k=0):
        self.na[k] += na
        self.nb[k] += nb
        self.nboth[k] += nboth
        self.nunion[k] += nunion

    def add_mask_groups(self, na, nb, nboth, nunion):
        self.na += na
        self.nb += nb
        self.nboth += nboth
        self.nunion += nunion

    def result(self, k=0):
        n = self.n[k]
        out = {'n_pixels': int(n),
               'n_new_valid': int(self.na[k]), 'n_ref_valid': int(self.nb[k]),
               'mask_jaccard': (float(self.nboth[k] / self.nunion[k])
                                if self.nunion[k] else None)}
        if n == 0:
            return out
        ma, mb = self.sa[k] / n, self.sb[k] / n
        ss_res = self.sd2[k]
        ss_tot = self.sb2[k] - n * mb * mb
        ss_a = self.sa2[k] - n * ma * ma
        cov = self.sab[k] - n * ma * mb
        rng = float(self.bmax[k] - self.bmin[k]) or 1.0
        out.update(
            r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            pearson=(float(cov / math.sqrt(ss_a * ss_tot))
                     if ss_a > 0 and ss_tot > 0 else None),
            rmse=float(math.sqrt(ss_res / n)),
            mae=float(self.sad[k] / n),
            mean_bias=float(self.sd[k] / n),
            max_abs_diff=float(self.maxad[k]),
            new_mean=float(ma), ref_mean=float(mb),
            ref_std=float(math.sqrt(max(ss_tot, 0) / n)),
            mean_abs_ref=float(self.sab_abs[k] / n),
            ref_range=rng,
        )
        if self.hist is not None:
            out['pct_within_1pct_range'] = float(
                100.0 * _count_below(self.hist[k], 0.01 * rng) / n)
            out['pct_exact'] = float(100.0 * self.hist[k][0] / n)
        return out


class TopK:
    """Keep the k largest |d| entries: (absd, lat, lon, key, new, ref)."""

    def __init__(self, k=10):
        self.k = k
        self.items = []

    def add(self, A, B, keys, lat, lon):
        D = np.abs(A.astype('float64') - B.astype('float64'))
        D = np.where(np.isfinite(D), D, -1.0)
        flat = D.ravel()
        k = min(self.k, flat.size)
        if k == 0:
            return
        idx = np.argpartition(flat, -k)[-k:]
        for i in idx:
            if flat[i] <= 0:
                continue
            t, r, c = np.unravel_index(i, D.shape)
            self.items.append((float(flat[i]), float(lat[r]), float(lon[c]),
                               key_str(keys[t]), float(A[t, r, c]),
                               float(B[t, r, c])))
        self.items.sort(key=lambda x: -x[0])
        del self.items[self.k:]


# ---------------------------------------------------------------------------
# block averaging (for 1/10 resolution maps)
# ---------------------------------------------------------------------------

def block_sum(arr, f):
    """Sum 2-D `arr` over f x f blocks (pads with 0)."""
    ny, nx = arr.shape
    py, px = (-ny) % f, (-nx) % f
    if py or px:
        arr = np.pad(arr, ((0, py), (0, px)))
    return arr.reshape(arr.shape[0] // f, f, arr.shape[1] // f, f).sum(axis=(1, 3))


class CoarseMean:
    """Accumulate the mean of per-pixel values on a grid coarsened by f."""

    def __init__(self, ny, nx, f=10):
        self.f = f
        self.s = np.zeros((math.ceil(ny / f), math.ceil(nx / f)))
        self.c = np.zeros_like(self.s)

    def add(self, row0, vals_sum, counts):
        """row0 = offset of the chunk in the (cropped) grid, multiple of f."""
        assert row0 % self.f == 0
        bs = block_sum(vals_sum, self.f)
        bc = block_sum(counts.astype('float64'), self.f)
        r = row0 // self.f
        self.s[r:r + bs.shape[0]] += bs
        self.c[r:r + bc.shape[0]] += bc

    def mean(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.c > 0, self.s / self.c, np.nan)


# ---------------------------------------------------------------------------
# netCDF writer (layout of step3_ensemble_prediction.py products)
# ---------------------------------------------------------------------------

def product_attrs(flux, version='v2.1', extra=None):
    meta = FLUX_METADATA.get(flux, {'long_name': flux, 'units': ''})
    attrs = {
        'crs': 'EPSG:4326', 'short_name': flux,
        'long_name': meta['long_name'], 'units': meta['units'],
        'version': version,
        'description': (
            f'AusEFlux {meta["long_name"]} is created by empirically upscaling '
            'the OzFlux eddy covariance network using machine learning methods '
            'coupled with climate and remote sensing datasets. The estimates '
            'provided within this dataset were extracted from an ensemble of '
            'predictions and represent the median and uncertainty range.'),
    }
    if extra:
        attrs.update(extra)
    return attrs


def create_nc(path, var_names, lat, lon, times, attrs=None, var_attrs=None,
              chunk=(1, 200, 200)):
    """
    Create a NETCDF4 file with dims (time, latitude, longitude); `lat`/`lon`
    keep their dtype; `times` are datetime-likes written as days since
    1970-01-01.  Returns the open netCDF4.Dataset (caller closes).
    """
    import netCDF4 as nc4
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds = nc4.Dataset(path, 'w', format='NETCDF4')
    ds.createDimension('time', len(times))
    ds.createDimension('latitude', len(lat))
    ds.createDimension('longitude', len(lon))
    tv = ds.createVariable('time', 'f8', ('time',))
    tv.units = 'days since 1970-01-01'
    tv.calendar = 'proleptic_gregorian'
    epoch = pd.Timestamp('1970-01-01')
    tv[:] = np.array([(pd.Timestamp(t) - epoch) / pd.Timedelta(days=1)
                      for t in times], dtype='float64')
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lv = ds.createVariable('latitude', lat.dtype.str.lstrip('<>='), ('latitude',))
    lv.units = 'degrees_north'
    lv[:] = lat
    xv = ds.createVariable('longitude', lon.dtype.str.lstrip('<>='), ('longitude',))
    xv.units = 'degrees_east'
    xv[:] = lon
    for vn in var_names:
        v = ds.createVariable(vn, 'f4', ('time', 'latitude', 'longitude'),
                              fill_value=np.float32(np.nan), zlib=True,
                              complevel=4,
                              chunksizes=(chunk[0], min(chunk[1], len(lat)),
                                          min(chunk[2], len(lon))))
        for k, val in (var_attrs or {}).get(vn, {}).items():
            if k in ('_FillValue',) or (isinstance(val, float) and np.isnan(val)):
                continue
            v.setncattr(k, val)
    for k, val in (attrs or {}).items():
        if isinstance(val, float) and np.isnan(val):
            continue
        if isinstance(val, (np.ndarray, list, tuple)) and len(val) == 0:
            continue
        ds.setncattr(k, val)
    return ds


def mid_month(year, month):
    return pd.Timestamp(year=int(year), month=int(month), day=15)


def need(path, what='file'):
    if not os.path.exists(path):
        sys.exit(f'ERROR: {what} not found: {path}')
