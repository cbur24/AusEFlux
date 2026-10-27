"""bitcmp.py A B [vars...]: bit-for-bit compare every {var}/{var}_500m_{y}.nc present in B (or listed) between dirs A and B."""
import sys, glob, os, numpy as np, netCDF4 as nc4
A, B = sys.argv[1], sys.argv[2]; only = sys.argv[3:]
ok = True; n = 0
for fb in sorted(glob.glob(f'{B}/*/*_500m_*.nc')):
    rel = os.path.relpath(fb, B); v = rel.split('/')[0]
    if only and v not in only: continue
    fa = f'{A}/{rel}'
    if not os.path.exists(fa): print('MISSING', fa); ok = False; continue
    with nc4.Dataset(fa) as a, nc4.Dataset(fb) as b:
        a.set_auto_mask(False); b.set_auto_mask(False)
        same = all(np.array_equal(np.asarray(a[k][:]).view('u1'), np.asarray(b[k][:]).view('u1'))
                   for k in (v, 'time', 'latitude', 'longitude'))
    n += 1; ok &= same
    if not same: print('DIFF', rel)
print(f'{"BIT-IDENTICAL" if ok else "DIFFERENT"} ({n} files) {A} vs {B}')
