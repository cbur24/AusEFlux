import numpy as np, pandas as pd, xarray as xr
rng = np.random.default_rng(0)
t = pd.date_range('2002-01-15', periods=288, freq='MS') + pd.Timedelta(days=14)
a = rng.gamma(2, 30, (288, 200, 300)).astype('float32'); a[rng.random(a.shape) < 0.05] = np.nan
da = xr.DataArray(a, dims=('time','latitude','longitude'), coords={'time': t})
def run():
    r = {}
    r['min7'] = da.rolling(time=7, min_periods=1).min().values
    r['mean9'] = da.rolling(time=9, min_periods=1).mean().values
    for w in (3, 6, 12): r[f'sum{w}'] = da.rolling(time=w, min_periods=w).sum().values
    r['clim'] = da.sel(time=slice('2003','2022')).groupby('time.month').mean().values
    r['resmax'] = da.resample(time='YE').max().values
    return r
out = {}
for bn in (False, True):
    for fl in (False, True):
        with xr.set_options(use_bottleneck=bn, use_flox=fl):
            out[(bn, fl)] = run()
base = out[(False, False)]
for k, r in out.items():
    print('bottleneck=%s flox=%s' % k, {n: bool(np.array_equal(base[n], r[n], equal_nan=True)) for n in base})
