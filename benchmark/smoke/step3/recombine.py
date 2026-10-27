# /// script
# dependencies = ["numpy<2", "netCDF4"]
# ///
"""(d) Recombine dumped members with reference nearest-rank; compare to product."""
import glob, sys, numpy as np, netCDF4 as nc
d, prod = sys.argv[1], sys.argv[2]
xx = np.stack([nc.Dataset(f)['GPP'][:].filled(np.nan) for f in sorted(f for f in glob.glob(f'{d}/*.nc') if not __import__('re').search(r'_r\d+-\d+\.nc$', f))])
vc = (~np.isnan(xx)).sum(0); xs = np.sort(xx, axis=0); ok = True
for q, b in zip((.25, .5, .75), ('GPP_25th_percentile', 'GPP_median', 'GPP_75th_percentile')):
    r = np.take_along_axis(xs, np.round(q * (vc - 1)).astype(np.int64).clip(0)[None], 0)[0]
    r[vc < 3] = np.nan; p = nc.Dataset(prod)[b][:].filled(np.nan)
    ok &= np.array_equal(np.isnan(r), np.isnan(p)) and np.array_equal(r[~np.isnan(r)].view('u4'), p[~np.isnan(p)].view('u4'))
print(('PASS' if ok else 'FAIL'), '(d) recombined dumps (nearest) == product, bit-for-bit')
