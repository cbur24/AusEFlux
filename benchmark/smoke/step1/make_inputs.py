"""Build a tiny fake AusEFlux raw-input tree for step1 smoke tests."""
import os, sys
import numpy as np, pandas as pd, xarray as xr

root = sys.argv[1]
inp = f'{root}/input'; st = f'{inp}/static'
rng = np.random.default_rng(42)

def mk(path, ds):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)

def centres(start, step, n):  # start = first centre
    return np.round(start + step * np.arange(n), 6)

# land mask 50x60 @0.005, descending lat
m_lat = centres(-20.0025, -0.005, 50); m_lon = centres(130.0025, 0.005, 60)
mask = np.ones((50, 60), bool); mask[:5, :8] = False   # a bit of "ocean"
mk(f'{st}/land_sea_mask_500m.nc', xr.Dataset({'landsea_mask_500m': (('latitude', 'longitude'), mask.astype('uint8'))},
   coords={'latitude': m_lat, 'longitude': m_lon}))

# smooth field helper: strong N-S gradient + E-W + noise
def field(lat, lon, t, amp=1.0, base=0.0, noise=0.05):
    L, O = np.meshgrid(lat, lon, indexing='ij')
    f = base + amp * ((L + 20.15) * 4 + 0.5 * np.sin((O - 130) * 20))
    f = f[None] * (1 + 0.1 * np.sin(np.arange(t) / 30.0))[:, None, None]
    return (f + noise * rng.standard_normal(f.shape)).astype('float32')

year = 2024
days = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
d8 = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='8D')

# MODIS 60x70 @0.005, descending (5-pixel margin around the mask)
x_lat = centres(-20.0025 + 5 * 0.005, -0.005, 60); x_lon = centres(130.0025 - 5 * 0.005, 0.005, 70)
mb = f'{inp}/MODIS/mosaic/MCD43A4.006'
for band, name, base in [('b01', '500m_0620_0670nm_nbar', 0.1), ('b02', '500m_0841_0876nm_nbar', 0.35),
                         ('b05', '500m_1230_1250nm_nbar', 0.25)]:
    v = np.abs(field(x_lat, x_lon, len(days), amp=0.05, base=base, noise=0.02))
    mk(f'{mb}/MCD43A4.006.{band}.{name}.{year}.nc',
       xr.Dataset({name: (('time', 'latitude', 'longitude'), v)},
                  coords={'time': days, 'latitude': x_lat, 'longitude': x_lon}))
lb = f'{inp}/MODIS/mosaic/MYD11A1.006'
lst = field(x_lat, x_lon, len(days), amp=5, base=300, noise=1)
mk(f'{lb}/MYD11A1.006.LST_Day_1km.{year}.nc', xr.Dataset({'LST_Day_1km': (('time', 'latitude', 'longitude'), lst)},
   coords={'time': days, 'latitude': x_lat, 'longitude': x_lon}))
qc = rng.choice(np.array([0, 5, 17, 65, 2, 3, 128], 'uint8'), size=lst.shape)
mk(f'{lb}/MYD11A1.006.QC_Day.{year}.nc', xr.Dataset({'QC_Day': (('time', 'latitude', 'longitude'), qc)},
   coords={'time': days, 'latitude': x_lat, 'longitude': x_lon}))

# OzWALD 8-day stored (time, longitude, latitude), descending lat, same 500 m grid
for var, base in [('NDVI', 0.5), ('LAI', 1.5)]:
    v = field(x_lat, x_lon, len(d8), amp=0.2, base=base, noise=0.02).transpose(0, 2, 1)
    mk(f'{inp}/OzWALD/8day/{var}/OzWALD.{var}.{year}.nc', xr.Dataset({var: (('time', 'longitude', 'latitude'), v)},
       coords={'time': d8, 'latitude': x_lat, 'longitude': x_lon}))

# coarse grids @0.05: OzWALD meteo descending, SILO ascending
c_lat_desc = centres(-19.875, -0.05, 12); c_lon = centres(129.875, 0.05, 14)
for var, base, amp in [('Tmin', 15, 3), ('Tmax', 30, 3), ('kTavg', 0.5, 0.05)]:
    v = field(c_lat_desc, c_lon, len(days), amp=amp, base=base, noise=0.01 * amp)
    mk(f'{inp}/OzWALD/daily/meteo/{var}/OzWALD.{var}.{year}.nc', xr.Dataset({var: (('time', 'latitude', 'longitude'), v)},
       coords={'time': days, 'latitude': c_lat_desc, 'longitude': c_lon}))
c_lat_asc = c_lat_desc[::-1]
for y in (year - 1, year):
    dd = pd.date_range(f'{y}-01-01', f'{y}-12-31', freq='D')
    specs = [('daily_rain', 'daily_rain', 3, 2)] if y == year - 1 else \
            [('daily_rain', 'daily_rain', 3, 2), ('radiation', 'radiation', 20, 3), ('vp', 'vp', 15, 2)]
    for folder, var, base, amp in specs:
        v = np.abs(field(c_lat_asc, c_lon, len(dd), amp=amp, base=base, noise=0.01 * amp))
        ds = xr.Dataset({var: (('time', 'lat', 'lon'), v), 'crs': ((), np.int32(0))},
                        coords={'time': dd, 'lat': c_lat_asc, 'lon': c_lon})
        mk(f'{inp}/SILO/{folder}/{y}.{var}.nc', ds)

# VegH 1 km @0.01, descending
v_lat = centres(-19.985, -0.01, 30); v_lon = centres(129.985, 0.01, 35)
mk(f'{st}/VegH_1km_2007_2010.nc', xr.Dataset({'VegH': (('latitude', 'longitude'), field(v_lat, v_lon, 1, amp=5, base=10, noise=0.2)[0])},
   coords={'latitude': v_lat, 'longitude': v_lon}))
print('inputs written under', root)
