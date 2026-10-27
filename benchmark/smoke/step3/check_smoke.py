# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2", "netCDF4"]
# ///
"""Bit-level checks of the smoke outputs (argv[1] = smoke dir)."""
import os, sys, glob
import numpy as np, netCDF4 as nc4
S = sys.argv[1]; P = 'GPP/AusEFlux_GPP_500m_quantiles_2024_v2.1.nc'
BANDS = ['GPP_25th_percentile', 'GPP_median', 'GPP_75th_percentile']
fails = 0

def load(path, names):
    with nc4.Dataset(path) as d:
        d.set_auto_mask(False)
        return {n: np.asarray(d[n][:]) for n in names + ['time', 'latitude', 'longitude']}, \
               {k: d.getncattr(k) for k in d.ncattrs()}

def same(a, b):  # bit-identical incl. NaN positions and dtype
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.isnan(a), np.isnan(b)) and np.array_equal(a.view(np.uint8), b.view(np.uint8)) \
        if a.dtype.kind == 'f' else np.array_equal(a, b)

CONFIG = ('quantile_method', 'mask_vars', 'mask_test', 'n_models', 'valid_only', 'lat_rows')

def strip(attrs, ignore):
    return {k: v for k, v in attrs.items() if k not in ignore}

# valid_only/all_pixels are bit-identical by design but recorded in the
# 'valid_only' attribute, so it is ignored when comparing the two modes.
def cmp(label, f1, f2, names=BANDS, ignore=('valid_only',)):
    global fails
    (a, aa), (b, ba) = load(f1, names), load(f2, names)
    bad = [k for k in a if not same(a[k], b[k])] + (['attrs'] if strip(aa, ignore) != strip(ba, ignore) else [])
    fails += bool(bad)
    print(f"{'PASS' if not bad else 'FAIL'} {label}" + (f'  differs: {bad}' if bad else ''))

o = f'{S}/out'
for m in ('linear', 'nearest'):
    cmp(f'(a) {m}: all_pixels == valid_only', f'{o}/{m}_all/{P}', f'{o}/{m}_valid/{P}')
    cmp(f'(b) {m}: months 1-6 + 7-12 + merge == unsharded', f'{o}/{m}_all/{P}', f'{o}/{m}_shard/{P}')
cmp('(a) nearest: TILE_ROWS=7 valid_only == all_pixels', f'{o}/nearest_all/{P}', f'{o}/nearest_tiles7/{P}')
cmp('(a) quantile script: all_pixels == valid_only', f'{o}/q_all/{P}', f'{o}/q_valid/{P}')
cmp('(b) quantile script: sharded+merge == unsharded', f'{o}/q_all/{P}', f'{o}/q_shard/{P}')
parts = sorted(glob.glob(f'{o}/nearest_shard/GPP/parts/*_m??.nc'))
print(f"{'PASS' if len(parts) == 12 else 'FAIL'} parts kept after merge: {len(parts)}")

# (c) member dumps: existence, dtype, dims, NaN mask == product NaN mask
from netCDF4 import Dataset
dump = f'{S}/dumps/nearest_all/GPP'
import re
names = sorted(n for n in (os.path.basename(p)[:-3] for p in glob.glob(f'{dump}/*.nc'))
               if not re.search(r'_r\d+-\d+$', n))  # exclude --lat_rows dumps
ok = names == ['GPP_lgbm_1', 'GPP_lgbm_2', 'GPP_rf_1', 'GPP_rf_2']
prod, _ = load(f'{o}/nearest_all/{P}', BANDS)
# expected mask from inputs: any mask var NaN, or urban == 1
feat = f'{S}/root/features_500m'
exp = np.zeros_like(prod['GPP_median'], bool)
for v in ('VegH', 'NDWI', 'rain_anom', 'Tavg'):
    with Dataset(f'{feat}/{v}_500m.nc') as d:
        t = np.asarray(d['time'][:]); exp |= ~np.isfinite(np.asarray(d[v][-12:]))
with Dataset(f'{S}/root/input/static/urban_water_mask_500m.nc') as d:
    exp |= (np.asarray(d['urban_water_mask'][:]) == 1)[None]
members = []
for n in names:
    with Dataset(f'{dump}/{n}.nc') as d:
        d.set_auto_mask(False); v = d['GPP']
        ok &= v.dimensions == ('time', 'latitude', 'longitude') and v.dtype == np.float32 \
            and v.filters()['zlib']
        a = np.asarray(v[:]); members.append(a)
        ok &= np.array_equal(np.isnan(a), exp)
for m in ('nearest_valid', 'linear_all'):  # dumps identical across modes
    for n, a in zip(names, members):
        with Dataset(f'{S}/dumps/{m}/GPP/{n}.nc') as d:
            d.set_auto_mask(False); ok &= same(a, np.asarray(d['GPP'][:]))
sp = sorted(glob.glob(f'{S}/dumps/nearest_shard/GPP/parts/*.nc'))
ok &= len(sp) == 48 and os.path.basename(sp[0]) == 'GPP_lgbm_1_m01.nc'
for n, a in zip(names, members):  # sharded dumps == unsharded dumps
    for mth in range(12):
        with Dataset(f'{S}/dumps/nearest_shard/GPP/parts/{n}_m{mth+1:02d}.nc') as d:
            d.set_auto_mask(False); ok &= same(a[mth:mth+1], np.asarray(d['GPP'][:]))
fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (c) member dumps: 4 files, float32 (time,lat,lon) zlib, "
      f"NaN == nodata|urban ({exp.mean():.1%} masked), identical across modes/sharding")

# ---------------------------------------------------------------------------
# (e) latitude window --lat_rows 10:25
# ---------------------------------------------------------------------------
R0, R1 = 10, 25
WP = f'GPP/AusEFlux_GPP_500m_quantiles_2024_v2.1_r{R0}-{R1}.nc'

def cmp_win(label, full_f, win_f, names=BANDS, member=False, R0=R0, R1=R1,
            ignore=('valid_only',)):
    """win_f must equal rows R0:R1 of full_f bit-for-bit (data + lat coord),
    same encoding, same attrs plus auseflux_lat_rows (and lat_rows)."""
    global fails
    (a, aa), (b, ba) = load(full_f, names), load(win_f, names)
    aa, ba = strip(aa, ignore), strip(ba, ignore)
    bad = []
    for k in a:
        exp = a[k][..., R0:R1, :] if k in names else (a[k][R0:R1] if k == 'latitude' else a[k])
        if not same(exp, b[k]):
            bad.append(k)
    exp_attrs = {**aa, 'auseflux_lat_rows': f'{R0}:{R1}'}
    if 'lat_rows' in aa:
        exp_attrs['lat_rows'] = f'{R0}:{R1}'
    if ba != exp_attrs:
        bad.append('attrs')
    with nc4.Dataset(full_f) as d1, nc4.Dataset(win_f) as d2:
        for n in names:
            v1, v2 = d1[n], d2[n]
            if (v1.dtype, v1.dimensions, v1.filters(), v1._FillValue.tobytes(),
                    v1.ncattrs()) != (v2.dtype, v2.dimensions, v2.filters(),
                                      v2._FillValue.tobytes(), v2.ncattrs()):
                bad.append(f'{n}:encoding')
        if d2.dimensions['latitude'].size != R1 - R0:
            bad.append('latdim')
    fails += bool(bad)
    print(f"{'PASS' if not bad else 'FAIL'} {label}" + (f'  differs: {bad}' if bad else ''))

for m in ('linear', 'nearest'):
    for mode in ('all', 'valid'):
        cmp_win(f'(e) {m} {mode}: lat_rows 10:25 == rows 10:25 of full',
                f'{o}/{m}_{mode}/{P}', f'{o}/{m}_{mode}/{WP}')
        for n in names:
            cmp_win(f'(e) {m} {mode}: member dump {n}_r10-25 == rows of full dump',
                    f'{S}/dumps/{m}_{mode}/GPP/{n}.nc',
                    f'{S}/dumps/{m}_{mode}/GPP/{n}_r{R0}-{R1}.nc', names=['GPP'])
    cmp(f'(e) {m}: windowed months 1-6 + 7-12 + merge == windowed year file',
        f'{o}/{m}_valid/{WP}', f'{o}/{m}_wshard/{WP}')
    wp = sorted(glob.glob(f'{o}/{m}_wshard/GPP/parts/*_m??_r{R0}-{R1}.nc'))
    wd = sorted(glob.glob(f'{S}/dumps/{m}_wshard/GPP/parts/*_m??_r{R0}-{R1}.nc'))
    ok = len(wp) == 12 and len(wd) == 48
    for n in names:  # windowed sharded dumps == windowed unsharded dumps
        with Dataset(f'{S}/dumps/{m}_valid/GPP/{n}_r{R0}-{R1}.nc') as d:
            d.set_auto_mask(False); full = np.asarray(d['GPP'][:])
        for mth in range(12):
            with Dataset(f'{S}/dumps/{m}_wshard/GPP/parts/{n}_m{mth+1:02d}_r{R0}-{R1}.nc') as d:
                d.set_auto_mask(False); ok &= same(full[mth:mth+1], np.asarray(d['GPP'][:]))
                ok &= d.getncattr('auseflux_lat_rows') == f'{R0}:{R1}'
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} (e) {m}: 12 windowed parts + 48 windowed part dumps, == windowed dumps")
cmp_win('(e) nearest TILE_ROWS=7 windowed valid_only == rows of full',
        f'{o}/nearest_all/{P}', f'{o}/nearest_tiles7/{WP}')
cmp_win('(e) quantile script all_pixels: lat_rows == rows of full', f'{o}/q_all/{P}', f'{o}/q_all/{WP}')
cmp_win('(e) quantile script valid_only: lat_rows == rows of full', f'{o}/q_valid/{P}', f'{o}/q_valid/{WP}')
cmp('(e) quantile script: windowed sharded+merge == windowed year file', f'{o}/q_all/{WP}', f'{o}/q_wshard/{WP}')
for tag in ('', 'q_'):
    st = open(f'{o}/{tag}merge_refuse.status').read().strip()
    log = open(f'{o}/{tag}merge_refuse.log').read()
    full_absent = not os.path.exists(f'{o}/{"q_wshard" if tag else "nearest_wshard"}/{P}')
    ok = st.startswith('refuse-merge: exit') and 'cannot be merged into a full product' in log and full_absent
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} (e) {tag or 'ensemble '}merge without --lat_rows refuses windowed parts ({st})")

# ---------------------------------------------------------------------------
# (f) new defaults (nearest, LAI_anom mask, --mask_test nan, valid_only)
# ---------------------------------------------------------------------------
cmp('(f) default == explicit nearest+LAI_anom+nan+valid_only', f'{o}/def_default/{P}', f'{o}/def_explicit/{P}')
cmp('(f) explicit reference config: valid_only == all_pixels', f'{o}/def_explicit/{P}', f'{o}/def_explicit_all/{P}')
cmp('(f) quantile script: default == explicit LAI_anom+nan+valid_only', f'{o}/q_def_default/{P}', f'{o}/q_def_explicit/{P}')
cmp('(f) quantile script: explicit reference mask valid_only == all_pixels', f'{o}/q_def_explicit/{P}', f'{o}/q_def_explicit_all/{P}')
prev = f'{S}/out_prev'
for lab, new, old in (('ensemble linear+VegH..+nonfinite+all_pixels', f'{o}/def_old/{P}', f'{prev}/linear_all/{P}'),
                      ('quantile VegH..+nonfinite+all_pixels', f'{o}/q_def_old/{P}', f'{prev}/q_all/{P}')):
    if os.path.exists(old):
        cmp(f'(f) explicit old defaults ({lab}) == saved old-default output (data; config attrs are new)',
            new, old, ignore=CONFIG + ('method',))
    else:
        print(f'SKIP (f) {lab}: no saved old-default output at {old}')

def differs(label, f1, f2):
    global fails
    (a, _), (b, _) = load(f1, BANDS), load(f2, BANDS)
    d = not all(same(a[k], b[k]) for k in BANDS)
    na, nb = int(np.isnan(a['GPP_median']).sum()), int(np.isnan(b['GPP_median']).sum())
    fails += not d
    print(f"{'PASS' if d else 'FAIL'} {label} (NaN median: {na} vs {nb})")
    return na, nb
differs('(f) new default differs from old default (LAI_anom mask, nearest)', f'{o}/def_default/{P}', f'{o}/def_old/{P}')
# +-Inf injected into mask vars LAI_anom/NDWI
cmp('(f) Inf features: default (valid_only, nan) == all_pixels', f'{o}/inf_default/{P}', f'{o}/inf_all/{P}')
na, nb = differs('(f) Inf features: --mask_test nan differs from nonfinite', f'{o}/inf_default/{P}', f'{o}/inf_nonfinite/{P}')
ok = na < nb; fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (f) Inf features: nan test predicts more pixels than nonfinite")
cmp('(f) Inf features, quantile script: default == all_pixels', f'{o}/q_inf_default/{P}', f'{o}/q_inf_all/{P}')
na, nb = differs('(f) Inf features, quantile script: nan differs from nonfinite', f'{o}/q_inf_default/{P}', f'{o}/q_inf_nonfinite/{P}')
ok = na < nb; fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (f) Inf features, quantile script: nan test predicts more pixels")

# ---------------------------------------------------------------------------
# review round 2: (g) windowed features, (h) time axis, (i) exit status,
# (j) merge_partial / config refusal, (k) argument checks, (l) attributes,
# (m) data bit-identical to the outputs of the code before this round
# ---------------------------------------------------------------------------
def status(n):
    return int(open(f'{o}/{n}.status').read().strip().split()[-1])

def expect(n, rc_ok, text=None, label=''):
    global fails
    rc = status(n); log = open(f'{o}/{n}.log').read()
    ok = (rc == 0) if rc_ok else (rc != 0)
    if text:
        ok &= text in log
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {label} [{n}: exit {rc}]" + ('' if ok or not text else f'  (expected text: {text!r})'))

W1025 = 'GPP/AusEFlux_GPP_500m_quantiles_2024_v2.1_r10-25.nc'
W1220 = 'GPP/AusEFlux_GPP_500m_quantiles_2024_v2.1_r12-20.nc'
expect('winfeat', True, label='(g) windowed features (attr 10:25), no --lat_rows: runs')
cmp_win('(g) windowed features -> _r10-25 product == rows 10:25 of full default product',
        f'{o}/def_default/{P}', f'{o}/winfeat/{W1025}', R0=10, R1=25)
cmp('(g) windowed features: all_pixels == valid_only', f'{o}/winfeat/{W1025}', f'{o}/winfeat_all/{W1025}')
cmp_win('(g) quantile script: windowed features -> _r10-25 == rows 10:25 of full',
        f'{o}/q_def_default/{P}', f'{o}/q_winfeat/{W1025}', R0=10, R1=25)
cmp_win('(g) windowed features + --lat_rows 12:20 (full-grid rows) == rows 12:20 of full',
        f'{o}/def_default/{P}', f'{o}/winfeat_sub/{W1220}', R0=12, R1=20)
cmp_win('(g) quantile script: windowed features + --lat_rows 12:20 == rows 12:20 of full',
        f'{o}/q_def_default/{P}', f'{o}/q_winfeat_sub/{W1220}', R0=12, R1=20)
with Dataset(f'{S}/root/input/static/urban_water_mask_500m.nc') as d:
    urb = np.asarray(d['urban_water_mask'][:]) == 1
wm, _ = load(f'{o}/winfeat/{W1025}', BANDS)
# smoke urban pixels lie in rows 20-24: the wrong (unshifted) slice 0:15 has none
ok = bool(urb[10:25].any()) and bool(np.isnan(wm['GPP_median'][:, urb[10:25]]).all()) \
    and not np.array_equal(urb[10:25], urb[0:15])
fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (g) windowed features: urban rows 10:25 masked (not rows 0:15)")
for t in ('', 'q_'):
    expect(f'{t}winfeat_outside', False, 'is not inside the latitude window 10:25', f'(g) {t}--lat_rows 5:20 outside feature window 10:25 refused')
    expect(f'{t}winfeat_noattr', False, 'does not match the feature grid', f'(g) {t}windowed features without attribute: urban-mask coordinate check refuses')
    expect(f'{t}tbad', False, 'time axis of feature B', f'(h) {t}time-axis mismatch (feature B) refused')
for t in ('winfeat_noattr', 'q_winfeat_noattr', 'tbad', 'q_tbad', 'winfeat_outside'):
    ok = not glob.glob(f'{o}/{t}/**/*.nc', recursive=True); fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} (g/h) {t}: no output written")
expect('nothing', False, 'requested outputs were NOT produced', '(i) empty models folder -> non-zero')
expect('nothing_nofolder', False, 'requested outputs were NOT produced', '(i) missing models folder -> non-zero')
expect('q_nothing', False, 'requested outputs were NOT produced', '(i) quantile: missing models folder -> non-zero')
expect('nomonth', False, 'no time step for month 01 in 2023', '(i) requested months without data -> non-zero')
ok = len(glob.glob(f'{o}/nomonth_dummy/GPP/parts/*.nc')) == 6; fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (i) ... but the 6 available months were still written")
expect('exists_ok', True, 'exists, skipping', '(i) existing product skipped -> exit 0')
ok = 'WARNING' not in open(f'{o}/exists_ok.log').read(); fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (i) same configuration: no WARNING")
expect('exists_warn', True, 'mask_test: file=nan requested=nonfinite', '(i) existing product with other config: exit 0 + WARNING naming the difference')
expect('q_exists_warn', True, 'valid_only: file=1 requested=0', '(i) quantile: existing product with other config: WARNING')
expect('exists_dumps', False, 'member dump(s) requested with --dump_members are missing', '(i) product exists but dumps missing -> non-zero, no silent skip')
expect('partial_strict', False, 'missing month part files', '(j) --merge with 3 of 12 parts refused')
expect('partial', True, 'months 4,5,6,7,8,9,10,11,12 missing', '(j) --merge_partial with 3 parts: exit 0 + WARNING')
expect('q_partial', True, 'months 4,5,6,7,8,9,10,11,12 missing', '(j) quantile --merge_partial: exit 0 + WARNING')
for pre, ref in (('', 'def_default'), ('q_', 'q_def_default')):
    (a, aa), (b, ba) = load(f'{o}/{ref}/{P}', BANDS), load(f'{o}/{pre}partial/{P}', BANDS)
    ok = all(same(a[k][:3], b[k]) for k in BANDS) and same(a['time'][:3], b['time']) \
        and ba.get('auseflux_partial_months') == '1,2,3' and strip(ba, ('auseflux_partial_months',)) == aa
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} (j) {pre}merge_partial product: 3 time steps == months 1-3 of full product, auseflux_partial_months=1,2,3")
expect('mix_merge', False, 'valid_only: 1 vs 0', '(j) merge refuses parts with different valid_only')
expect('q_mix_merge', False, 'mask_test: nan vs nonfinite', '(j) quantile merge refuses parts with different mask_test')
for t in ('mix', 'q_mix'):
    ok = not os.path.exists(f'{o}/{t}/{P}') and not glob.glob(f'{o}/{t}/GPP/*.tmp'); fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} (j) {t}: no product and no .tmp left")
expect('bad_nmodels', False, '--n_models must be >= 1', '(k) --n_models 0 rejected')
expect('bad_threads', False, '--threads must be', '(k) --threads 0 rejected')
expect('q_bad_threads', False, '--threads must be', '(k) quantile --threads 0 rejected')
expect('bad_months', False, 'error: invalid --months', '(k) --months 1-13 -> parser error')
_, at = load(f'{o}/def_default/{P}', BANDS)
exp = {'quantile_method': 'nearest', 'mask_vars': 'LAI_anom,VegH,NDWI,rain_anom,Tavg', 'mask_test': 'nan',
       'n_models': 4, 'valid_only': 1, 'lat_rows': 'all'}
ok = all(str(at.get(k)) == str(v) for k, v in exp.items())
_, dt = load(f'{S}/dumps/nearest_all/GPP/GPP_rf_1.nc', ['GPP'])
_, pt = load(f'{o}/nearest_shard/GPP/parts/AusEFlux_GPP_500m_quantiles_2024_v2.1_m03.nc', BANDS)
_, qt = load(f'{o}/q_def_default/{P}', BANDS)
ok &= all(k in dt and k in pt for k in CONFIG) and dt['valid_only'] == 0 and pt['quantile_method'] == 'nearest'
ok &= {k: str(qt.get(k)) for k in ('mask_vars', 'mask_test', 'valid_only', 'lat_rows')} == \
    {'mask_vars': 'LAI_anom,VegH,NDWI,rain_anom,Tavg', 'mask_test': 'nan', 'valid_only': '1', 'lat_rows': 'all'}
_, wt = load(f'{o}/winfeat/{W1025}', BANDS)
ok &= wt.get('lat_rows') == '10:25' and wt.get('auseflux_lat_rows') == '10:25'
fails += not ok
print(f"{'PASS' if ok else 'FAIL'} (l) configuration attributes on products, parts, dumps, quantile products, windowed products")
left = glob.glob(f'{o}/**/*.tmp', recursive=True) + glob.glob(f'{S}/dumps/**/*.tmp', recursive=True)
fails += bool(left)
print(f"{'PASS' if not left else 'FAIL'} no .tmp files left anywhere ({len(left)})")
base = f'{S}/out_review2_before'
if os.path.isdir(base):
    n = bad = 0
    for fb in sorted(glob.glob(f'{base}/**/*.nc', recursive=True)):
        fa = f'{o}/{os.path.relpath(fb, base)}'
        if not os.path.exists(fa):
            continue
        with nc4.Dataset(fb) as d:
            vs = [v for v in d.variables]
        (a, _), (b, _) = load(fa, [v for v in vs if v not in ('time', 'latitude', 'longitude')]), \
            load(fb, [v for v in vs if v not in ('time', 'latitude', 'longitude')])
        n += 1
        if set(a) != set(b) or not all(same(a[k], b[k]) for k in a):
            bad += 1; print('   differs:', os.path.relpath(fb, base))
    fails += bad > 0 or n < 30
    print(f"{'PASS' if bad == 0 and n >= 30 else 'FAIL'} (m) {n} outputs data/coords bit-identical to the pre-change run (TILE_ROWS 1500->1400 etc.)")
else:
    print('SKIP (m): no out_review2_before')

print('ALL PASS' if fails == 0 else f'{fails} FAILURES')
sys.exit(1 if fails else 0)
