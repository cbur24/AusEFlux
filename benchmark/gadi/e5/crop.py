"""crop.py IN OUT R0:R1 -- copy latitude rows of an AusEFlux product (all vars)."""
import sys
import xarray as xr
src, dst, rows = sys.argv[1:4]
r0, r1 = map(int, rows.split(':'))
ds = xr.open_dataset(src)
ds.isel(latitude=slice(r0, r1)).load().to_netcdf(dst)
print('wrote', dst)
