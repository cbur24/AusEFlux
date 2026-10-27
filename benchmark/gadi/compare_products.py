# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "xarray", "netCDF4", "pandas", "scipy", "matplotlib", "rasterio"]
# ///
"""
Streamed comparison of two AusEFlux quantile products (NEW vs REF) on the
continental 500 m grid, with a tolerance scorecard.

Products are read in latitude chunks (all aligned months of ~400 rows at a
time; never a whole band) and every statistic is accumulated in float64.

Alignment
  * time: on (year, month) after xarray decode (15th-of-month vs other
    encodings both work);
  * space: coordinates matched within 1e-4 deg, cropped to the overlap
    (orientation may differ; error if the grid spacing differs);
    --border N drops an N-pixel border of the overlap.

Per band (25th_percentile, median, 75th_percentile; or --bands median), over
jointly-valid pixel-months:
  n_pix, R2, RMSE, MAE, mean bias, max|d|, % within 1 % of the reference
  range, mask Jaccard -- overall, per month and per Major Vegetation Group
  (MVG raster nearest-sampled onto the product grid once, cached as .npy in
  --out); continental monthly mean series of both products (area-weighted by
  cos(lat); over each product's own valid pixels and over joint pixels);
  annual sum per pixel (sum of the aligned months, pixels valid in all of
  them) with continental and per-MVG area-weighted totals and relative
  differences; and a tolerance scorecard (thresholds from the plan's
  Tolerance section, overridable with --thresholds JSON).

"% within 1 % range" uses a 0.0005-decade log histogram of |d| (exact zeros
counted exactly; the bin holding the threshold is interpolated), so it is
exact to well below 0.01 percentage points.

Outputs: {out}/metrics.json, {out}/metrics.md and, with --plots,
{out}/plots/{band}_{diffmap,scatter,monthly}.png.

Examples
    uv run compare_products.py NEW.nc REF.nc --flux GPP --out cmp/GPP_2025 --plots
    uv run compare_products.py NEW.nc REF.nc --flux NEE --bands median --border 5
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (BANDS, CoarseMean, Field, GridMap, GroupStats,  # noqa: E402
                     align_time, cell_area_km2, choose_rows, fmt, key_str,
                     log, md_table, parse_months, write_json)

DEFAULT_MVG = '/g/data/xc0/project/AusEFlux/data/aus6_0e_mvg.tif'

# NVIS 6.0 Major Vegetation Groups
MVG_NAMES = {
    0: 'outside MVG raster / nodata',
    1: 'Rainforests and Vine Thickets', 2: 'Eucalypt Tall Open Forests',
    3: 'Eucalypt Open Forests', 4: 'Eucalypt Low Open Forests',
    5: 'Eucalypt Woodlands', 6: 'Acacia Forests and Woodlands',
    7: 'Callitris Forests and Woodlands', 8: 'Casuarina Forests and Woodlands',
    9: 'Melaleuca Forests and Woodlands', 10: 'Other Forests and Woodlands',
    11: 'Eucalypt Open Woodlands', 12: 'Tropical Eucalypt Woodlands/Grasslands',
    13: 'Acacia Open Woodlands', 14: 'Mallee Woodlands and Shrublands',
    15: 'Low Closed Forests and Tall Closed Shrublands', 16: 'Acacia Shrublands',
    17: 'Other Shrublands', 18: 'Heathlands', 19: 'Tussock Grasslands',
    20: 'Hummock Grasslands',
    21: 'Other Grasslands, Herblands, Sedgelands and Rushlands',
    22: 'Chenopod Shrublands, Samphire Shrublands and Forblands',
    23: 'Mangroves', 24: 'Inland Aquatic', 25: 'Cleared, non-native vegetation, buildings',
    26: 'Unclassified native vegetation', 27: 'Naturally bare',
    28: 'Sea and estuaries', 29: 'Regrowth, modified native vegetation',
    30: 'Unclassified Forest', 31: 'Other Open Woodlands',
    32: 'Mallee Open Woodlands and Sparse Mallee Shrublands',
    99: 'Unknown/no data',
}

DEFAULT_THRESHOLDS = {
    'r2_min': {'GPP': 0.99, 'ET': 0.99, 'NEE': 0.97, 'ER': 0.97, 'default': 0.97},
    'abs_bias_max_frac_of_mean_abs_ref': 0.01,
    'rmse_max_frac_of_ref_std': 0.05,
    'mask_jaccard_min': 0.999,
    'monthly_mean_max_rel': 0.02,
    # 'month': |new-ref| / |ref| of that month; 'mean_abs': divide by the mean
    # |ref| of the monthly series (robust for NEE months near zero)
    'monthly_rel_denominator': 'month',
    'annual_total_max_rel': 0.01,
    'annual_mvg_max_rel': 0.02,
    'mvg_min_area_frac': 0.001,      # gate only MVGs covering >= 0.1 % of area
    'mvg_exclude': [0, 28, 99],
}

UNIT_SCALE = 1e-6   # (g/m2 or mm) * km2 -> TgC or km3


def load_thresholds(path):
    th = copy.deepcopy(DEFAULT_THRESHOLDS)
    if path:
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(th.get(k), dict):
                th[k].update(v)
            else:
                th[k] = v
    return th


# ---------------------------------------------------------------------------
# MVG raster on the product grid (cached)
# ---------------------------------------------------------------------------

def mvg_on_grid(tif, lat, lon, out_dir, rows=400):
    ny, nx = len(lat), len(lon)
    tag = f'mvg_{ny}x{nx}_{lat[0]:.5f}_{lon[0]:.5f}_{lat[-1]:.5f}_{lon[-1]:.5f}.npy'
    cache = os.path.join(out_dir, tag)
    if os.path.exists(cache):
        log(f'MVG: using cache {cache}')
        return np.load(cache)
    if not tif or not os.path.exists(tif):
        log(f'MVG: raster not found ({tif}); per-MVG breakdown skipped')
        return None
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    dlat = abs(float(np.median(np.diff(lat)))) if ny > 1 else 0.005
    dlon = abs(float(np.median(np.diff(lon)))) if nx > 1 else 0.005
    west = float(lon.min()) - dlon / 2
    north = float(lat.max()) + dlat / 2
    tr = from_origin(west, north, dlon, dlat)
    out = np.zeros((ny, nx), dtype='uint8')
    t0 = time.time()
    with rasterio.open(tif) as src:
        nodata = src.nodata
        with WarpedVRT(src, crs='EPSG:4326', transform=tr, width=nx, height=ny,
                       resampling=Resampling.nearest,
                       src_nodata=nodata, nodata=0) as vrt:
            for r in range(0, ny, rows):
                h = min(rows, ny - r)
                blk = vrt.read(1, window=Window(0, r, nx, h))
                blk = np.where((blk < 0) | (blk > 255), 0, blk)
                out[r:r + h] = blk.astype('uint8')
    # VRT rows are north->south, cols west->east; match the product order
    if ny > 1 and lat[0] < lat[-1]:
        out = out[::-1]
    if nx > 1 and lon[0] > lon[-1]:
        out = out[:, ::-1]
    out = np.ascontiguousarray(out)
    os.makedirs(out_dir, exist_ok=True)
    np.save(cache, out)
    log(f'MVG: sampled onto grid in {time.time() - t0:.1f}s -> {cache}')
    return out


# ---------------------------------------------------------------------------
# per-band streaming state
# ---------------------------------------------------------------------------

class BandState:
    def __init__(self, T, ny, nx, with_mvg):
        self.overall = GroupStats(1)
        self.month = GroupStats(T)
        self.mvg = GroupStats(256, hist=False) if with_mvg else None
        self.annual = GroupStats(1)
        # monthly area-weighted means
        z = lambda: np.zeros(T)  # noqa: E731
        self.own_new_s, self.own_new_w = z(), z()
        self.own_ref_s, self.own_ref_w = z(), z()
        self.joint_new_s, self.joint_ref_s, self.joint_w = z(), z(), z()
        # annual totals (sum annual * km2) and areas
        self.tot = {k: 0.0 for k in ('joint_new', 'joint_ref', 'joint_area',
                                     'own_new', 'own_new_area',
                                     'own_ref', 'own_ref_area')}
        self.n_pix_annual = 0
        self.mvg_new = np.zeros(256)
        self.mvg_ref = np.zeros(256)
        self.mvg_area = np.zeros(256)
        self.diffmap = CoarseMean(ny, nx, 10)
        self.samples = []


def process_chunk(st, A, B, w_rows, area_rows, mv, y_off, p_sample, rng):
    T = A.shape[0]
    va = np.isfinite(A)
    vb = np.isfinite(B)
    both = va & vb
    na_t = va.sum(axis=(1, 2)).astype('float64')
    nb_t = vb.sum(axis=(1, 2)).astype('float64')
    nboth_t = both.sum(axis=(1, 2)).astype('float64')
    nun_t = na_t + nb_t - nboth_t
    st.overall.add_mask(na_t.sum(), nb_t.sum(), nboth_t.sum(), nun_t.sum())
    st.month.add_mask_groups(na_t, nb_t, nboth_t, nun_t)

    a = A[both]
    b = B[both]
    tl = np.repeat(np.arange(T), nboth_t.astype(np.int64))
    st.overall.add(a, b)
    st.month.add(a, b, tl)
    if st.mvg is not None:
        mvl = mv.astype(np.int64)
        st.mvg.add(a, b, np.broadcast_to(mvl, A.shape)[both])
        fl = mvl.ravel()
        cnt = lambda m: np.bincount(fl, weights=m.sum(0).ravel(),  # noqa: E731
                                    minlength=256)[:256]
        ca, cb, cab = cnt(va), cnt(vb), cnt(both)
        st.mvg.add_mask_groups(ca, cb, cab, ca + cb - cab)
    if p_sample > 0 and a.size:
        k = rng.binomial(a.size, min(1.0, p_sample))
        if k:
            idx = rng.integers(0, a.size, k)
            st.samples.append(np.stack([a[idx], b[idx]], axis=1).astype('float32'))
    del a, b, tl

    # monthly area-weighted means (row sums first, then cos(lat) weights)
    Az = np.where(va, A, 0)
    Bz = np.where(vb, B, 0)
    st.own_new_s += Az.sum(axis=2, dtype='float64') @ w_rows
    st.own_new_w += va.sum(axis=2).astype('float64') @ w_rows
    st.own_ref_s += Bz.sum(axis=2, dtype='float64') @ w_rows
    st.own_ref_w += vb.sum(axis=2).astype('float64') @ w_rows
    Aj = np.where(both, A, 0)
    Bj = np.where(both, B, 0)
    st.joint_new_s += Aj.sum(axis=2, dtype='float64') @ w_rows
    st.joint_ref_s += Bj.sum(axis=2, dtype='float64') @ w_rows
    st.joint_w += both.sum(axis=2).astype('float64') @ w_rows

    # annual sums per pixel (pixels valid in every aligned month)
    annA = Az.sum(axis=0, dtype='float64')
    annB = Bz.sum(axis=0, dtype='float64')
    area = np.broadcast_to(area_rows[:, None], annA.shape)
    allj = both.all(axis=0)
    alla = va.all(axis=0)
    allb = vb.all(axis=0)
    st.annual.add(annA[allj], annB[allj])
    st.annual.add_mask(alla.sum(), allb.sum(), allj.sum(), (alla | allb).sum())
    st.n_pix_annual += int(allj.sum())
    st.tot['joint_new'] += float((annA * area)[allj].sum())
    st.tot['joint_ref'] += float((annB * area)[allj].sum())
    st.tot['joint_area'] += float(area[allj].sum())
    st.tot['own_new'] += float((annA * area)[alla].sum())
    st.tot['own_new_area'] += float(area[alla].sum())
    st.tot['own_ref'] += float((annB * area)[allb].sum())
    st.tot['own_ref_area'] += float(area[allb].sum())
    if st.mvg is not None:
        lab = mv[allj].astype(np.int64)
        st.mvg_new += np.bincount(lab, weights=(annA * area)[allj], minlength=256)[:256]
        st.mvg_ref += np.bincount(lab, weights=(annB * area)[allj], minlength=256)[:256]
        st.mvg_area += np.bincount(lab, weights=area[allj], minlength=256)[:256]

    # 1/10-resolution mean difference map
    st.diffmap.add(y_off, Aj.sum(axis=0, dtype='float64') - Bj.sum(axis=0, dtype='float64'),
                   both.sum(axis=0))


def rel(n, r):
    return (n - r) / abs(r) if r not in (0, None) and r == r else None


def finalize(st, keys, flux, th, mvg_present):
    res = {'overall': st.overall.result(0)}
    res['per_month'] = [dict(month=key_str(k), **st.month.result(i))
                        for i, k in enumerate(keys)]
    series = []
    for i, k in enumerate(keys):
        on = st.own_new_s[i] / st.own_new_w[i] if st.own_new_w[i] else None
        orf = st.own_ref_s[i] / st.own_ref_w[i] if st.own_ref_w[i] else None
        jn = st.joint_new_s[i] / st.joint_w[i] if st.joint_w[i] else None
        jr = st.joint_ref_s[i] / st.joint_w[i] if st.joint_w[i] else None
        series.append(dict(month=key_str(k), new_own=on, ref_own=orf,
                           new_joint=jn, ref_joint=jr,
                           rel_diff_joint=rel(jn, jr) if jn is not None else None,
                           rel_diff_own=rel(on, orf) if on is not None else None))
    res['monthly_series'] = series
    tot = st.tot
    ann = {'months_summed': len(keys), 'pixel_stats': st.annual.result(0),
           'n_pixels': st.n_pix_annual,
           'unit': 'km3/yr' if flux == 'ET' else 'TgC/yr',
           'total_joint_new': tot['joint_new'] * UNIT_SCALE,
           'total_joint_ref': tot['joint_ref'] * UNIT_SCALE,
           'area_joint_km2': tot['joint_area'],
           'total_own_new': tot['own_new'] * UNIT_SCALE,
           'total_own_ref': tot['own_ref'] * UNIT_SCALE,
           'area_own_new_km2': tot['own_new_area'],
           'area_own_ref_km2': tot['own_ref_area']}
    ann['rel_diff_joint'] = rel(tot['joint_new'], tot['joint_ref'])
    ann['rel_diff_own'] = rel(tot['own_new'], tot['own_ref'])
    res['annual'] = ann
    if mvg_present and st.mvg is not None:
        rows = []
        total_area = st.mvg_area.sum()
        for g in range(256):
            m = st.mvg.result(g)
            if m['n_pixels'] == 0 and m['n_new_valid'] == 0 and m['n_ref_valid'] == 0:
                continue
            area = st.mvg_area[g]
            rows.append(dict(
                mvg=g, name=MVG_NAMES.get(g, f'class {g}'), **m,
                annual_area_km2=area,
                annual_area_frac=area / total_area if total_area else None,
                annual_total_new=st.mvg_new[g] * UNIT_SCALE,
                annual_total_ref=st.mvg_ref[g] * UNIT_SCALE,
                annual_mean_new=st.mvg_new[g] / area if area else None,
                annual_mean_ref=st.mvg_ref[g] / area if area else None,
                annual_rel_diff=rel(st.mvg_new[g], st.mvg_ref[g]) if area else None))
        res['per_mvg'] = rows
    res['scorecard'] = scorecard(res, flux, th)
    return res


def scorecard(res, flux, th):
    ov = res['overall']
    checks = []

    def add(name, value, thr, ok):
        checks.append(dict(check=name, value=value, threshold=thr,
                           pass_=None if value is None else bool(ok)))

    r2min = th['r2_min'].get(flux, th['r2_min'].get('default', 0.97))
    r2 = ov.get('r2')
    add('R2 >= min', r2, r2min, r2 is not None and r2 >= r2min)
    if ov.get('n_pixels'):
        fb = abs(ov['mean_bias']) / ov['mean_abs_ref'] if ov['mean_abs_ref'] else None
        add('abs(bias) / mean abs(ref) <=', fb, th['abs_bias_max_frac_of_mean_abs_ref'],
            fb is not None and fb <= th['abs_bias_max_frac_of_mean_abs_ref'])
        fr = ov['rmse'] / ov['ref_std'] if ov['ref_std'] else None
        add('RMSE / std(ref) <=', fr, th['rmse_max_frac_of_ref_std'],
            fr is not None and fr <= th['rmse_max_frac_of_ref_std'])
    jac = ov.get('mask_jaccard')
    add('mask Jaccard >=', jac, th['mask_jaccard_min'],
        jac is not None and jac >= th['mask_jaccard_min'])
    ser = [s for s in res['monthly_series'] if s['ref_joint'] is not None]
    if ser:
        if th.get('monthly_rel_denominator') == 'mean_abs':
            den = np.mean([abs(s['ref_joint']) for s in ser]) or None
            vals = [abs(s['new_joint'] - s['ref_joint']) / den for s in ser] if den else []
        else:
            vals = [abs(s['rel_diff_joint']) for s in ser
                    if s['rel_diff_joint'] is not None]
        mx = max(vals) if vals else None
        add('max monthly-mean rel diff <=', mx, th['monthly_mean_max_rel'],
            mx is not None and mx <= th['monthly_mean_max_rel'])
    a = res['annual'].get('rel_diff_joint')
    add('annual total rel diff <=', None if a is None else abs(a),
        th['annual_total_max_rel'],
        a is not None and abs(a) <= th['annual_total_max_rel'])
    if 'per_mvg' in res:
        el = [g for g in res['per_mvg']
              if g['mvg'] not in th['mvg_exclude']
              and (g['annual_area_frac'] or 0) >= th['mvg_min_area_frac']
              and g['annual_rel_diff'] is not None]
        mx = max((abs(g['annual_rel_diff']) for g in el), default=None)
        worst = max(el, key=lambda g: abs(g['annual_rel_diff']))['name'] if el else None
        add(f'max per-MVG annual rel diff <= (worst: {worst})', mx,
            th['annual_mvg_max_rel'], mx is not None and mx <= th['annual_mvg_max_rel'])
    decided = [c['pass_'] for c in checks if c['pass_'] is not None]
    return {'checks': checks, 'pass': bool(decided) and all(decided)}


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#2b2b29', '#6b6a64', '#e1e0d9'


def make_plots(outdir, flux, band, st, grid, keys, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False})
    label = f'{flux}_{band}'
    lat, lon = grid.lat, grid.lon
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    # difference map (1/10 resolution), diverging blue - gray - red
    dm = st.diffmap.mean()
    if lat[0] < lat[-1]:
        dm = dm[::-1]
    if lon[0] > lon[-1]:
        dm = dm[:, ::-1]
    cmap = LinearSegmentedColormap.from_list(
        'div', ['#104281', '#3987e5', '#f0efec', '#e66767', '#a32828'])
    cmap.set_bad('white')
    finite = dm[np.isfinite(dm)]
    v = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    v = v or 1.0
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(dm, extent=extent, origin='upper', cmap=cmap, vmin=-v, vmax=v,
                   interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f'mean monthly new - ref ({flux})')
    ax.set_title(f'{label}: mean difference (1/10 resolution)', color=INK)
    ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{band}_diffmap.png'), dpi=120)
    plt.close(fig)

    # density scatter
    if st.samples:
        s = np.concatenate(st.samples)
        if len(s) > 1_000_000:
            s = s[np.random.default_rng(1).choice(len(s), 1_000_000, replace=False)]
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        hb = ax.hexbin(s[:, 1], s[:, 0], gridsize=120, mincnt=1, bins='log',
                       cmap='Blues')
        lo = float(min(s.min(), 0)); hi = float(s.max())
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1)
        fig.colorbar(hb, ax=ax, label='pixel-months (log)')
        ov = res['overall']
        ax.set_title(f'{label}  R²={fmt(ov.get("r2"), 5)}  n_sample={len(s)}', color=INK)
        ax.set_xlabel('reference'); ax.set_ylabel('new')
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{band}_scatter.png'), dpi=120)
        plt.close(fig)

    # monthly continental means
    ser = res['monthly_series']
    x = np.arange(len(ser))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for key, col, mk, name in (('ref_joint', ORANGE, 's', 'reference'),
                               ('new_joint', BLUE, 'o', 'new')):
        y = [np.nan if r[key] is None else r[key] for r in ser]
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6, label=name)
    ax.set_xticks(x, [r['month'] for r in ser], rotation=45, ha='right')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_ylabel(f'area-weighted mean {flux}')
    ax.set_title(f'{label}: continental monthly mean (joint valid pixels)', color=INK)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{band}_monthly.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def write_md(path, meta, results):
    L = [f'# compare_products: {meta["flux"]}', '',
         f'- new: `{meta["new"]}`', f'- ref: `{meta["ref"]}`',
         f'- aligned months: {", ".join(meta["months"])}',
         f'- only in new: {meta["only_new"] or "none"}; only in ref: {meta["only_ref"] or "none"}',
         f'- overlap grid: {meta["grid"]["lat"]["n"]} x {meta["grid"]["lon"]["n"]} '
         f'(border {meta["border"]}), max coord offset '
         f'{fmt(meta["grid"]["lat"]["max_coord_offset"], 6)} / '
         f'{fmt(meta["grid"]["lon"]["max_coord_offset"], 6)} deg',
         f'- wall time: {meta["wall_s"]:.1f} s', '']
    L += ['## Overall (jointly valid pixel-months)', '',
          md_table(['band', 'n_pix', 'R²', 'RMSE', 'MAE', 'mean bias', 'max abs Δ',
                    '% within 1% range', '% exact', 'mask Jaccard', 'scorecard'],
                   [[b, r['overall'].get('n_pixels'), r['overall'].get('r2'),
                     r['overall'].get('rmse'), r['overall'].get('mae'),
                     r['overall'].get('mean_bias'), r['overall'].get('max_abs_diff'),
                     r['overall'].get('pct_within_1pct_range'),
                     r['overall'].get('pct_exact'),
                     r['overall'].get('mask_jaccard'), r['scorecard']['pass']]
                    for b, r in results.items()]), '']
    for b, r in results.items():
        L += [f'## {b}', '', '### Scorecard', '',
              md_table(['check', 'value', 'threshold', 'result'],
                       [[c['check'], c['value'], c['threshold'],
                         'n/a' if c['pass_'] is None else c['pass_']]
                        for c in r['scorecard']['checks']]), '']
        a = r['annual']
        L += ['### Annual sums', '',
              f'Sum of {a["months_summed"]} aligned months per pixel; '
              f'continental totals in {a["unit"]} (area-weighted).', '',
              md_table(['pixels', 'total new', 'total ref', 'rel diff', 'R² (pixel)',
                        'own-valid total new', 'own-valid total ref', 'rel diff (own)'],
                       [[a['n_pixels'], a['total_joint_new'], a['total_joint_ref'],
                         a['rel_diff_joint'], a['pixel_stats'].get('r2'),
                         a['total_own_new'], a['total_own_ref'], a['rel_diff_own']]]), '']
        L += ['### Per month', '',
              md_table(['month', 'n_pix', 'R²', 'RMSE', 'bias', 'max abs Δ', 'Jaccard',
                        'mean new', 'mean ref', 'rel diff'],
                       [[m['month'], m.get('n_pixels'), m.get('r2'), m.get('rmse'),
                         m.get('mean_bias'), m.get('max_abs_diff'), m.get('mask_jaccard'),
                         s['new_joint'], s['ref_joint'], s['rel_diff_joint']]
                        for m, s in zip(r['per_month'], r['monthly_series'])]), '']
        if 'per_mvg' in r:
            L += ['### Per Major Vegetation Group', '',
                  md_table(['MVG', 'name', 'n_pix', 'R²', 'RMSE', 'bias', 'Jaccard',
                            'area km²', 'annual new', 'annual ref', 'annual rel diff'],
                           [[g['mvg'], g['name'], g.get('n_pixels'), g.get('r2'),
                             g.get('rmse'), g.get('mean_bias'), g.get('mask_jaccard'),
                             g['annual_area_km2'], g['annual_total_new'],
                             g['annual_total_ref'], g['annual_rel_diff']]
                            for g in r['per_mvg']]), '']
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('new', help='new product (AusEFlux_{flux}_500m_quantiles_{year}_v2.1.nc)')
    ap.add_argument('ref', help='reference product')
    ap.add_argument('--flux', required=True, help='GPP, NEE, ER or ET (variable prefix)')
    ap.add_argument('--bands', choices=['all', 'median'], default='all',
                    help='all = 25th/median/75th (default), median = median only')
    ap.add_argument('--mvg', default=DEFAULT_MVG,
                    help=f'Major Vegetation Group GeoTIFF (default {DEFAULT_MVG}); '
                         'skipped with a message if missing; "" disables')
    ap.add_argument('--out', default='compare_out', help='output directory')
    ap.add_argument('--plots', action='store_true', help='write PNG diagnostics')
    ap.add_argument('--border', type=int, default=0,
                    help='exclude an N-pixel border of the overlap')
    ap.add_argument('--months', default=None,
                    help='restrict to these calendar months, e.g. "1-6"')
    ap.add_argument('--rows', type=int, default=400,
                    help='latitude rows per chunk (default 400; raised to the '
                         'file chunk height, rounded to a multiple of 10)')
    ap.add_argument('--thresholds', default=None,
                    help='JSON file overriding tolerance thresholds, e.g. '
                         '{"r2_min": {"NEE": 0.98}, "mask_jaccard_min": 0.9995}')
    ap.add_argument('--pixel_area', choices=['sphere', '500m'], default='sphere',
                    help='cell area for totals: sphere = R²·dφ·dλ·cos(lat) from '
                         'the grid spacing (default); 500m = 0.25 km²·cos(lat)')
    ap.add_argument('--samples', type=int, default=1_000_000,
                    help='random pixel-month sample size for the scatter plot')
    args = ap.parse_args()

    t_start = time.time()
    flux = args.flux
    bands = BANDS if args.bands == 'all' else ['median']
    th = load_thresholds(args.thresholds)
    os.makedirs(args.out, exist_ok=True)
    months = parse_months(args.months)

    newf = {b: Field(args.new, f'{flux}_{b}') for b in bands}
    reff = {b: Field(args.ref, f'{flux}_{b}') for b in bands}
    nf0, rf0 = newf[bands[0]], reff[bands[0]]
    grid = GridMap(nf0, rf0, border=args.border)
    keys, only_new, only_ref = align_time(
        nf0, rf0, (lambda k: k[1] in months) if months else None)
    if not keys:
        sys.exit('ERROR: no common (year, month) between products')
    years = sorted({k[0] for k in keys})
    if len(years) > 1:
        log(f'WARNING: aligned months span several years {years}; '
            'annual sums are over all of them')
    T = len(keys)
    ny, nx = grid.y.n, grid.x.n
    log(f'{flux}: {T} months, overlap {ny} x {nx}, bands {bands}')

    mvg = mvg_on_grid(args.mvg, grid.lat, grid.lon, args.out) if args.mvg else None
    lat = grid.lat
    dlat = float(np.median(np.diff(nf0.lat))) if len(nf0.lat) > 1 else 0.005
    dlon = float(np.median(np.diff(nf0.lon))) if len(nf0.lon) > 1 else 0.005
    w_all = np.cos(np.deg2rad(lat))
    area_all = cell_area_km2(lat, dlat, dlon, args.pixel_area)

    rows = choose_rows(args.rows, list(newf.values()) + list(reff.values()), 10)
    total_pm = T * ny * nx
    p_sample = min(1.0, 2.5 * args.samples / total_pm) if args.plots else 0.0
    rng = np.random.default_rng(12345)
    states = {b: BandState(T, ny, nx, mvg is not None) for b in bands}
    xn, xr_ = grid.xsel()
    nchunks = math.ceil(ny / rows)
    for ci, (a, b_, (ysn, ysr)) in enumerate(grid.row_chunks(rows)):
        y_off = a - grid.y.n0
        sl = slice(y_off, y_off + (b_ - a))
        mv = mvg[sl] if mvg is not None else None
        for band in bands:
            A = newf[band].read(keys, ysn, xn)
            B = reff[band].read(keys, ysr, xr_)
            process_chunk(states[band], A, B, w_all[sl], area_all[sl], mv, y_off,
                          p_sample, rng)
            del A, B
        log(f'  chunk {ci + 1}/{nchunks} rows {a}-{b_} '
            f'({time.time() - t_start:.0f}s)')

    results = {b: finalize(states[b], keys, flux, th, mvg is not None) for b in bands}
    meta = dict(flux=flux, new=os.path.abspath(args.new), ref=os.path.abspath(args.ref),
                months=[key_str(k) for k in keys],
                only_new=[key_str(k) for k in only_new],
                only_ref=[key_str(k) for k in only_ref],
                grid=grid.to_dict(), border=args.border, rows_per_chunk=rows,
                pixel_area=args.pixel_area, thresholds=th,
                wall_s=time.time() - t_start)
    if args.plots:
        for b in bands:
            make_plots(os.path.join(args.out, 'plots'), flux, b, states[b], grid,
                       keys, results[b])
    meta['wall_s'] = time.time() - t_start
    write_json(os.path.join(args.out, 'metrics.json'), dict(meta=meta, bands=results))
    write_md(os.path.join(args.out, 'metrics.md'), meta, results)
    for b in bands:
        ov = results[b]['overall']
        log(f'{flux}_{b}: n={ov.get("n_pixels")} R2={fmt(ov.get("r2"), 6)} '
            f'RMSE={fmt(ov.get("rmse"))} bias={fmt(ov.get("mean_bias"))} '
            f'max|d|={fmt(ov.get("max_abs_diff"))} jac={fmt(ov.get("mask_jaccard"), 6)} '
            f'scorecard={"PASS" if results[b]["scorecard"]["pass"] else "FAIL"}')
    log(f'wrote {args.out}/metrics.json, metrics.md ({meta["wall_s"]:.1f}s)')


if __name__ == '__main__':
    main()
