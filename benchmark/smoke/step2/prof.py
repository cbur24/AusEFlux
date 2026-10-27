import os
import sys, time, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('s2', os.path.join(os.environ.get('REPO', '.'), 'step2_feature_datasets.py')); s2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2)
d = sys.argv[1]
cfg = s2.Config(d+'/interim', d+'/prof_out', d+'/static', reuse_clim=False)
for var in ['NDWI', 'SRAD']:
    src = s2._GapfillSource(cfg, var)
    tr = 50; tread = tcomp = 0
    for s in range(0, src.n_lat, tr):
        t0 = time.time(); tile = src.da.isel(latitude=slice(s, s+tr)).compute(scheduler='synchronous'); t1 = time.time()
        src.compute(s, s+tr)  # includes read again
        t2 = time.time(); tread += t1-t0; tcomp += (t2-t1) - (t1-t0)
    print(var, f'read {tread:.2f}s compute {tcomp:.2f}s')
    out = s2._OutFile(d+'/prof_tmp.nc', 'x', src.lat, src.lon, src.times.values)
    arr = np.random.rand(len(src.times), tr, src.n_lon).astype('float32')
    t0 = time.time()
    for s in range(0, src.n_lat, tr): out.write(s, s+tr, arr)
    out.finish(); print(var, f'write {time.time()-t0:.2f}s')
