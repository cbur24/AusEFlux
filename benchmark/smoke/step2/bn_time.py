import time, numpy as np, pandas as pd, xarray as xr
rng = np.random.default_rng(0)
t = pd.date_range('2002-01-01', periods=288, freq='MS') + pd.Timedelta(days=14)
a = rng.gamma(2, 30, (288, 200, 1000)).astype('float32'); a[rng.random(a.shape) < 0.05] = np.nan
da = xr.DataArray(a, dims=('time','latitude','longitude'), coords={'time': t})
ops = {
 'rolling mean9': lambda: da.rolling(time=9, min_periods=1).mean().values,
 'rolling sum12': lambda: da.rolling(time=12, min_periods=12).sum().values,
 'groupby clim':  lambda: da.sel(time=slice('2003','2022')).groupby('time.month').mean().values,
 'groupby fillna': lambda: da.groupby('time.month').fillna(da.groupby('time.month').mean()).values,
}
for name, f in ops.items():
    res = {}
    for acc in (False, True):
        with xr.set_options(use_bottleneck=acc, use_flox=acc):
            t0 = time.time(); res[acc] = f(); dt = time.time() - t0
        print(f'{name:15s} accel={acc!s:5s} {dt:6.2f}s')
    x, y = res[False].astype('f8'), res[True].astype('f8')
    rel = np.nanmax(np.abs(x - y) / np.maximum(np.abs(x), 1e-12))
    print(f'{"":15s} max rel diff {rel:.2e}, differing {np.mean(~((x == y) | (np.isnan(x) & np.isnan(y)))):.3%}')
