import os
"""Atomic-write / race / label-pairing checks for step1 (argv: smoke_step1 dir, work dir)."""
import sys, os, glob, shutil, subprocess
import numpy as np, pandas as pd, xarray as xr, netCDF4 as nc4
S, W = sys.argv[1], os.path.abspath(sys.argv[2])
sys.path.insert(0, os.environ.get('REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
import step1_harmonisation as s1
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
INP = f'{S}/data/input'; s1.set_paths(INP, None)
mask = xr.open_dataset(s1.static_files()['land_mask'])['landsea_mask_500m']
tmpl = mask.rio.write_crs('EPSG:4326')
fails = 0
def check(ok, label):
    global fails; fails += not ok; print(('PASS ' if ok else 'FAIL ') + label)
def bits(a, b, v):
    with nc4.Dataset(a) as x, nc4.Dataset(b) as y:
        x.set_auto_mask(False); y.set_auto_mask(False)
        return all(np.array_equal(np.asarray(x[k][:]).view('u1'), np.asarray(y[k][:]).view('u1'))
                   for k in (v, 'time', 'latitude', 'longitude'))
R = W + '/res/'
ndwi = f'{R}NDWI/NDWI_500m_2024.nc'
# 1. truncated legacy output is regenerated, not skipped
os.makedirs(os.path.dirname(ndwi)); open(ndwi, 'wb').write(b'CDF\x01garbage')
s1._modis_indices('NDWI', '2024', R, tmpl, mask, verbose=True)
check(bits(ndwi, f'{S}/out_default/NDWI/NDWI_500m_2024.nc', 'NDWI') and not s1._part_files(ndwi),
      'truncated legacy NDWI regenerated, bit-identical to out_default, no part file')
# 2. exception mid-write: no final file, no part file
kf = f'{R}kNDVI/kNDVI_500m_2024.nc'; orig = s1._write_slice; n = [0]
def boom(*a):
    n[0] += 1
    if n[0] == 3: raise RuntimeError('simulated crash')
    orig(*a)
s1._write_slice = boom
try:
    s1._modis_indices('kNDVI', '2024', R, tmpl, mask)
    crashed = False
except RuntimeError:
    crashed = True
s1._write_slice = orig
check(crashed and not os.path.exists(kf) and not s1._part_files(kf),
      'crash during write leaves neither final nor .part file')
# 3. race: final appears while writing -> second writer discards its copy
rf = f'{R}rain/rain_500m_2023.nc'
s1._SILO_climate('2023', R, tmpl, mask, prior_year_rain_only=True)   # first writer (rain_prior)
first = open(rf, 'rb').read(); ino = os.stat(rf).st_ino
lat, lon = s1._template_coords(tmpl)
with s1._nc_writer(rf, 'rain', lat, lon) as d:      # second writer (SILO 2023 shard)
    s1._write_slice(d, 'rain', 0, pd.Timestamp('2023-01-15'), np.zeros((len(lat), len(lon)), 'f4'))
check(os.stat(rf).st_ino == ino and open(rf, 'rb').read() == first and not s1._part_files(rf),
      'concurrent writer of rain_2023 discards its copy, first file kept, no part file')
s1._SILO_climate('2023', R, tmpl, mask, verbose=True, prior_year_rain_only=True)  # skip path
check(bits(rf, f'{S}/out_default/rain/rain_500m_2023.nc', 'rain'), 'rain_2023 bit-identical to out_default')
# 4. VegH time axis: NDWI with a .part sibling is not used
short = f'{W}/short/NDWI/NDWI_500m_2024.nc'; os.makedirs(os.path.dirname(short))
xr.open_dataarray(ndwi).isel(time=slice(0, 3)).to_netcdf(short)
check(len(s1._veg_time_axis('2024', f'{W}/short/')) == 3, 'VegH uses complete NDWI time axis (3 months)')
open(short + '.part.99999', 'w').close()
check(len(s1._veg_time_axis('2024', f'{W}/short/')) == 12,
      'VegH ignores NDWI with in-progress .part sibling -> MCD43A4 months (12)')
# 5. VPD pairs Tavg by label: reversed Tavg time order gives bit-identical VPD
for v in ('Tavg',):
    os.makedirs(f'{W}/rev/{v}'); os.makedirs(f'{W}/miss/{v}')
    t = xr.open_dataarray(f'{S}/out_default/Tavg/Tavg_500m_2024.nc').load()
    t.isel(time=slice(None, None, -1)).to_netcdf(f'{W}/rev/Tavg/Tavg_500m_2024.nc')
    t.isel(time=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]).to_netcdf(f'{W}/miss/Tavg/Tavg_500m_2024.nc')
s1._SILO_climate('2024', f'{W}/rev/', tmpl, mask)
check(bits(f'{W}/rev/VPD/VPD_500m_2024.nc', f'{S}/out_default/VPD/VPD_500m_2024.nc', 'VPD'),
      'VPD with time-reversed Tavg == out_default (label pairing)')
code = subprocess.run([sys.executable, '-c', f'''
import sys; sys.path.insert(0, os.environ.get("REPO", "."))
import step1_harmonisation as s1, xarray as xr
s1.set_paths("{INP}", None)
m = xr.open_dataset(s1.static_files()["land_mask"])["landsea_mask_500m"]
s1._SILO_climate("2024", "{W}/miss/", m.rio.write_crs("EPSG:4326"), m)'''], capture_output=True, text=True)
check(code.returncode != 0 and 'no time step 2024-04-15' in code.stdout
      and not glob.glob(f'{W}/miss/VPD/*'), f'Tavg missing April -> exit {code.returncode}, clear error, no VPD/part file')
print('ALL OK' if not fails else f'{fails} FAILURES')
