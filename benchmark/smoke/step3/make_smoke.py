# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy<2", "pandas", "xarray>=2025.1.0", "netCDF4", "joblib",
#                 "lightgbm", "scikit-learn==1.3.2"]
# ///
"""Build a synthetic AusEFlux tree for step3 smoke tests under argv[1]."""
import os, sys
import numpy as np, pandas as pd, xarray as xr, joblib
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

root = sys.argv[1]
feat_dir = f'{root}/features_500m'; static = f'{root}/input/static'
res = f'{root}/results'
for d in (feat_dir, static, f'{res}/models/ensemble/v2.1/GPP',
          f'{res}/models/quantile/v2.1/GPP'):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(0)
vars_ = ['A', 'B', 'C', 'VegH', 'NDWI', 'rain_anom', 'Tavg', 'LAI_anom']
with open(f'{res}/variables.txt', 'w') as f:
    f.write(','.join(f'{v}_EC' if v == 'VegH' else f'{v}_RS' for v in vars_) + ',site,\n')
ny, nx = 40, 50
lat = np.round(-10 - 0.005 * np.arange(ny), 4); lon = np.round(113 + 0.005 * np.arange(nx), 4)
# 2023-07..2024-12 so that .sel(time='2024') is exercised
time = pd.date_range('2023-07-01', '2024-12-01', freq='MS') + pd.Timedelta(days=14)
ocean = np.zeros((ny, nx), bool); ocean[:, :6] = True            # shared no-data
for v in vars_:
    a = rng.normal(0, 1, (len(time), ny, nx)).astype('float32')
    a[:, ocean] = np.nan
    a[rng.random(a.shape) < 0.03] = np.nan                         # sporadic NaN
    if v == 'A':
        a[rng.random(a.shape) < 0.02] = np.inf                     # non-finite in non-mask var
    xr.Dataset({v: (('time', 'latitude', 'longitude'), a)},
               coords={'time': time, 'latitude': lat, 'longitude': lon}
               ).to_netcdf(f'{feat_dir}/{v}_500m.nc')
urban = np.zeros((ny, nx), 'float32'); urban[20:25, 20:30] = 1
xr.Dataset({'urban_water_mask': (('latitude', 'longitude'), urban)},
           coords={'latitude': lat, 'longitude': lon}).to_netcdf(f'{static}/urban_water_mask_500m.nc')
X = rng.normal(0, 1, (2000, len(vars_))).astype('float32')
y = X[:, 0] * 2 + X[:, 3] - X[:, 6] ** 2 + rng.normal(0, 0.5, 2000)
md = f'{res}/models/ensemble/v2.1/GPP'
for i in (1, 2):
    joblib.dump(LGBMRegressor(n_estimators=50, random_state=i, verbose=-1).fit(X, y), f'{md}/GPP_lgbm_{i}.joblib')
    joblib.dump(RandomForestRegressor(n_estimators=20, max_depth=8, random_state=i).fit(X, y), f'{md}/GPP_rf_{i}.joblib')
qd = f'{res}/models/quantile/v2.1/GPP'
for qn, a in (('q25', .25), ('q50', .5), ('q75', .75)):
    joblib.dump(LGBMRegressor(objective='quantile', alpha=a, n_estimators=50, verbose=-1).fit(X, y), f'{qd}/GPP_{qn}.joblib')
print('smoke tree built at', root)
