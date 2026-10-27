#!/bin/bash
# Smoke tests for step2_feature_datasets.py (synthetic 40x50 grid).
set -u
S=${S:-$(cd "$(dirname "$0")" && pwd)}
R=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
STEP2="uv run $R/step2_feature_datasets.py"
CMP="uv run -q $S/compare.py"
BASE="--interim_path $S/data/interim --static_path $S/data/static"
# E5: defaults are now the reference semantics (--no_bare_clip --anom_guard none).
# The tests below were written for the pre-E5 defaults; OLD reproduces them.
OLD="--bare_clip --anom_guard one"
ARGS="$BASE $OLD"
cd $S
PASS=(); FAIL=()
ok()   { PASS+=("$1"); echo "PASS $1"; }
bad()  { FAIL+=("$1"); echo "FAIL $1"; }
run()  { local out=$1; shift; rm -rf $out; $STEP2 $ARGS --results_path $out "$@" > $out.log 2>&1 || { echo "run failed: $out"; tail -20 $out.log; }; }

# baseline: original script (git HEAD) without bottleneck/flox
rm -rf out_orig; uv run $S/step2_orig.py $BASE --results_path out_orig > out_orig.log 2>&1

# (a) default run == original, 28 outputs + climatologies
run out_def
n=$(ls out_def/*.nc | wc -l)
$CMP out_orig out_def && [ "$n" = 28 ] && ok "a: $OLD == HEAD script (old defaults), $n files" || bad "a: old defaults"

# (a2) tile size independence (uneven 7-row tiles)
run out_t7 --tile_rows 7
$CMP out_def out_t7 --recursive && ok "a2: --tile_rows 7 == auto" || bad "a2: tile_rows"

# (b) --no_bare_clip: negative bare somewhere, everything else identical
run out_nobare --no_bare_clip
uv run -q --with xarray --with netCDF4 python - <<EOF && ok "b: --no_bare_clip negative bare" || bad "b: no_bare_clip"
import xarray as xr, numpy as np, sys
b = xr.open_dataarray('$S/out_nobare/bare_500m.nc').values
c = xr.open_dataarray('$S/out_def/bare_500m.nc').values
t = xr.open_dataarray('$S/out_nobare/trees_500m.nc').values + xr.open_dataarray('$S/out_nobare/grass_500m.nc').values
print('  min bare unclipped', np.nanmin(b), 'clipped', np.nanmin(c), 'frac<0', np.mean(b < 0))
ref = (1 - t).astype('float32')
sys.exit(0 if (np.nanmin(b) < 0 and np.nanmin(c) >= 0
               and np.array_equal(np.maximum(b, 0), c, equal_nan=True)) else 1)
EOF
mkdir -p cmp_b1 cmp_b2; rm -f cmp_b1/* cmp_b2/*
for f in out_def/*.nc; do g=$(basename $f); [ "$g" = bare_500m.nc ] && continue; ln -s $S/$f cmp_b1/$g; ln -s $S/out_nobare/$g cmp_b2/$g; done
$CMP cmp_b1 cmp_b2 && ok "b2: other files unchanged by --no_bare_clip" || bad "b2"

# (c) --anom_guard none: NaN where 0/0 (rain Jan block), Inf where x/0 (Tavg Jan block)
run out_none --anom_guard none
uv run -q --with xarray --with netCDF4 python $S/check_c.py $S 2>/dev/null && ok "c: --anom_guard none == src/clim exactly (NaN/Inf kept), one == guarded" || bad "c: anom_guard"

# (d) --years_out 2005-2006, fresh dir (no stored climatologies, everything inline)
run out_y --years_out 2005-2006
n=$(ls out_y/*.nc | wc -l)
$CMP out_def out_y --years 2005-2006 && [ "$n" = 28 ] && ok "d1: --years_out fresh == slice of full ($n files)" || bad "d1: years_out fresh"
# (d2) sharded like a PBS array: gapfill per var, then each remaining stage separately
rm -rf out_ysh; mkdir -p out_ysh
for v in LAI LST NDVI NDWI SRAD Tavg VPD VegH kNDVI rain; do $STEP2 $ARGS --results_path out_ysh --years_out 2005-2006 --stages gapfill --vars $v >> out_ysh.log 2>&1; done
for st in c4 anom deltat cmlrain vegfrac c4; do $STEP2 $ARGS --results_path out_ysh --years_out 2005-2006 --stages $st >> out_ysh.log 2>&1; done
$CMP out_def out_ysh --years 2005-2006 && ok "d2: --years_out sharded stages == slice of full" || bad "d2: years_out sharded"
# (d3) annual update: stored climatologies from the full run reused (windowed reads)
rm -rf out_yc; mkdir -p out_yc; cp -r out_def/climatology out_yc/
$STEP2 $ARGS --results_path out_yc --years_out 2005-2006 > out_yc.log 2>&1
grep -c "reusing climatology" out_yc.log
$CMP out_def out_yc --years 2005-2006 && ok "d3: --years_out with reused climatologies == slice of full" || bad "d3"
# (d4) single year at the start of the record
run out_y03 --years_out 2003-2003
$CMP out_def out_y03 --years 2003-2003 && ok "d4: --years_out 2003-2003 == slice" || bad "d4"
# (d5) full-mode stage sharding (gapfill by var, anom by var) == default
rm -rf out_sh; mkdir -p out_sh
for v in LAI LST NDVI NDWI SRAD Tavg VPD VegH kNDVI rain; do $STEP2 $ARGS --results_path out_sh --stages gapfill --vars $v >> out_sh.log 2>&1; done
$STEP2 $ARGS --results_path out_sh --stages vegfrac,cmlrain,deltat >> out_sh.log 2>&1
for v in NDWI SRAD Tavg VPD kNDVI LAI rain rain_cml3 rain_cml6 rain_cml12; do $STEP2 $ARGS --results_path out_sh --stages anom --vars $v >> out_sh.log 2>&1; done
$STEP2 $ARGS --results_path out_sh --stages c4 >> out_sh.log 2>&1
$CMP out_def out_sh --recursive && ok "d5: sharded full run == default (incl. climatologies)" || bad "d5"

# (e) --threads 3 == --threads 1 (7-row tiles -> 6 tiles)
run out_th3 --threads 3 --tile_rows 7
$CMP out_t7 out_th3 --recursive && ok "e: --threads 3 (process pool) == --threads 1" || bad "e: threads"
run out_yth3 --threads 3 --tile_rows 7 --years_out 2005-2006
$CMP out_def out_yth3 --years 2005-2006 && ok "e2: --threads 3 --years_out == slice" || bad "e2"
run out_th3t --threads 3 --tile_rows 7 --pool thread
$CMP out_t7 out_th3t --recursive && ok "e3: --threads 3 --pool thread == --threads 1" || bad "e3: thread pool"

# (f) --reuse_clim: rerun with the stored climatologies (delete outputs only)
rm -rf out_rc; cp -r out_def out_rc; rm -f out_rc/*.nc
/usr/bin/time -f "%e s" $STEP2 $ARGS --results_path out_rc > out_rc.log 2>&1
grep -c "reusing climatology" out_rc.log
$CMP out_def out_rc --recursive && ok "f: --reuse_clim rerun == first run (incl. climatology files)" || bad "f: reuse_clim"
rm -rf out_nrc; cp -r out_def out_nrc; rm -f out_nrc/*.nc
$STEP2 $ARGS --results_path out_nrc --no_reuse_clim > out_nrc.log 2>&1
$CMP out_def out_nrc --recursive && ok "f2: --no_reuse_clim recompute == stored" || bad "f2"

# (e4) automatic tile height shared by threads (--max_tile_mem tiny -> 25-row floor, 2 tiles)
run out_th3m --threads 3 --max_tile_mem 0.001
grep -q "max_tile_mem=0.001GB" out_th3m.log && $CMP out_def out_th3m --recursive && ok "e4: --threads 3 --max_tile_mem 0.001 (auto 25-row tiles) == default" || bad "e4: max_tile_mem"

# (g) --lat_rows 10:25: every output == rows 10:25 of the full run, bit for bit
W="--lat_rows 10:25"; RW="--rows 10:25"
run out_r $W
n=$(ls out_r/*.nc | wc -l); nc=$(ls out_r/climatology/*_r10-25.nc | wc -l)
$CMP out_def out_r $RW --recursive && [ "$n" = 28 ] && ok "g1: --lat_rows 10:25 == rows of full ($n files, $nc window climatologies)" || bad "g1: lat_rows"
run out_r7 $W --tile_rows 4
$CMP out_def out_r7 $RW --recursive && ok "g1b: --lat_rows --tile_rows 4 == rows of full" || bad "g1b"
run out_ry $W --years_out 2005-2006
$CMP out_def out_ry $RW --years 2005-2006 && ok "g2: --lat_rows --years_out fresh (inline recompute) == rows+slice of full" || bad "g2"
run out_ry03 $W --years_out 2003-2003
$CMP out_def out_ry03 $RW --years 2003-2003 && ok "g2b: --lat_rows --years_out 2003-2003 == rows+slice" || bad "g2b"
run out_rth3 $W --threads 3 --tile_rows 4
$CMP out_def out_rth3 $RW --recursive && ok "g3: --lat_rows --threads 3 (4-row tiles) == rows of full" || bad "g3"
run out_rth3m $W --threads 3 --max_tile_mem 0.0001
$CMP out_def out_rth3m $RW --recursive && ok "g3b: --lat_rows --threads 3 auto tiles == rows of full" || bad "g3b"
run out_ryth3 $W --threads 3 --tile_rows 4 --years_out 2005-2006
$CMP out_def out_ryth3 $RW --years 2005-2006 && ok "g3c: --lat_rows --threads 3 --years_out == rows+slice" || bad "g3c"
# (g4) stored window climatologies reused (full rerun and annual update)
rm -rf out_rrc; mkdir -p out_rrc; cp -r out_r/climatology out_rrc/
$STEP2 $ARGS --results_path out_rrc $W > out_rrc.log 2>&1
k=$(grep -c "reusing climatology" out_rrc.log)
$CMP out_def out_rrc $RW --recursive && [ "$k" -gt 0 ] && ok "g4: --lat_rows reusing $k window climatologies == rows of full" || bad "g4"
rm -rf out_ryc; mkdir -p out_ryc; cp -r out_r/climatology out_ryc/
$STEP2 $ARGS --results_path out_ryc $W --years_out 2005-2006 > out_ryc.log 2>&1
k=$(grep -c "reusing climatology" out_ryc.log)
$CMP out_def out_ryc $RW --years 2005-2006 && [ "$k" -gt 0 ] && ok "g4b: --lat_rows --years_out reusing $k window climatologies == rows+slice" || bad "g4b"
# full-grid climatologies present in a windowed results dir are never used (and vice versa)
rm -rf out_rx; mkdir -p out_rx; cp -r out_def/climatology out_rx/
$STEP2 $ARGS --results_path out_rx $W > out_rx.log 2>&1
k=$(grep -c "reusing climatology" out_rx.log)
rm -rf cmp_rx; mkdir -p cmp_rx; cp out_rx/*.nc cmp_rx/; mkdir cmp_rx/climatology; cp out_rx/climatology/*_r10-25.nc cmp_rx/climatology/
$CMP out_def cmp_rx $RW --recursive && [ "$k" = 0 ] && ok "g4c: full-grid climatologies ignored by a --lat_rows run" || bad "g4c"
rm -rf out_xr; mkdir -p out_xr; cp -r out_r/climatology out_xr/
$STEP2 $ARGS --results_path out_xr > out_xr.log 2>&1
k=$(grep -c "reusing climatology" out_xr.log)
rm -f out_xr/climatology/*_r10-25.nc
$CMP out_def out_xr --recursive && [ "$k" = 0 ] && ok "g4d: window climatologies ignored by a full run" || bad "g4d"
# (g5) ablation switches
run out_rnobare $W --no_bare_clip
$CMP out_nobare out_rnobare $RW --recursive && ok "g5: --lat_rows --no_bare_clip == rows of --no_bare_clip full" || bad "g5"
run out_rnone $W --anom_guard none
$CMP out_none out_rnone $RW --recursive && ok "g5b: --lat_rows --anom_guard none == rows of --anom_guard none full" || bad "g5b"
# (g6) sharded by stages / vars
rm -rf out_rsh; mkdir -p out_rsh
for v in LAI LST NDVI NDWI SRAD Tavg VPD VegH kNDVI rain; do $STEP2 $ARGS --results_path out_rsh $W --stages gapfill --vars $v >> out_rsh.log 2>&1; done
$STEP2 $ARGS --results_path out_rsh $W --stages vegfrac,cmlrain,deltat >> out_rsh.log 2>&1
for v in NDWI SRAD Tavg VPD kNDVI LAI rain rain_cml3 rain_cml6 rain_cml12; do $STEP2 $ARGS --results_path out_rsh $W --stages anom --vars $v >> out_rsh.log 2>&1; done
$STEP2 $ARGS --results_path out_rsh $W --stages c4 >> out_rsh.log 2>&1
$CMP out_def out_rsh $RW --recursive && ok "g6: --lat_rows sharded stages/vars == rows of full" || bad "g6"
rm -rf out_rysh; mkdir -p out_rysh
for v in LAI LST NDVI NDWI SRAD Tavg VPD VegH kNDVI rain; do $STEP2 $ARGS --results_path out_rysh $W --years_out 2005-2006 --stages gapfill --vars $v >> out_rysh.log 2>&1; done
for st in c4 anom deltat cmlrain vegfrac c4; do $STEP2 $ARGS --results_path out_rysh $W --years_out 2005-2006 --stages $st >> out_rysh.log 2>&1; done
$CMP out_def out_rysh $RW --years 2005-2006 && ok "g6b: --lat_rows --years_out sharded (stages out of order) == rows+slice" || bad "g6b"
# (g7) window at the grid edges
run out_r0 --lat_rows 0:40
$CMP out_def out_r0 --rows 0:40 --recursive && ok "g7: --lat_rows 0:40 (whole grid) == full" || bad "g7"
run out_r39 --lat_rows 39:40
$CMP out_def out_r39 --rows 39:40 --recursive && ok "g7b: --lat_rows 39:40 (last row) == rows of full" || bad "g7b"
$STEP2 $ARGS --results_path out_rbad --lat_rows 30:41 > out_rbad.log 2>&1; rc=$?
[ $rc != 0 ] && grep -q "exceeds" out_rbad.log && ok "g7c: --lat_rows beyond the grid fails" || bad "g7c"

# (n) E5 defaults: no flags == --no_bare_clip --anom_guard none (reference semantics)
runb() { local out=$1; shift; rm -rf $out; $STEP2 $BASE --results_path $out "$@" > $out.log 2>&1 || { echo "run failed: $out"; tail -20 $out.log; }; }
runb out_new
runb out_newx --no_bare_clip --anom_guard none
grep -q "bare_clip=False anom_guard=none" out_new.log && $CMP out_newx out_new --recursive && ok "n1: default == --no_bare_clip --anom_guard none" || bad "n1: new defaults"
rm -rf cmp_n1 cmp_n2; mkdir -p cmp_n1 cmp_n2
for f in out_new/*.nc; do g=$(basename $f); case $g in bare_500m.nc) ln -s $S/out_nobare/$g cmp_n1/$g;; *_anom_500m.nc) ln -s $S/out_none/$g cmp_n1/$g;; *) ln -s $S/out_def/$g cmp_n1/$g;; esac; ln -s $S/$f cmp_n2/$g; done
$CMP cmp_n1 cmp_n2 && ok "n2: default == old run with bare from --no_bare_clip and anomalies from --anom_guard none" || bad "n2"
runb out_nr --lat_rows 10:25 --tile_rows 4 --threads 3 --years_out 2005-2006
$CMP out_new out_nr --rows 10:25 --years 2005-2006 && ok "n3: default --lat_rows 10:25 --tile_rows 4 --threads 3 --years_out == rows+slice of default" || bad "n3"
rm -rf out_nsh; mkdir -p out_nsh
for v in LAI,LST NDVI,NDWI SRAD,Tavg VPD,VegH kNDVI,rain; do $STEP2 $BASE --results_path out_nsh --lat_rows 5:30 --years_out 2005-2006 --stages gapfill --vars $v >> out_nsh.log 2>&1; done
$STEP2 $BASE --results_path out_nsh --lat_rows 5:30 --years_out 2005-2006 --stages vegfrac,cmlrain,deltat >> out_nsh.log 2>&1
for v in NDWI SRAD Tavg VPD kNDVI LAI rain rain_cml3 rain_cml6 rain_cml12; do $STEP2 $BASE --results_path out_nsh --lat_rows 5:30 --years_out 2005-2006 --stages anom --vars $v >> out_nsh.log 2>&1; done
$STEP2 $BASE --results_path out_nsh --lat_rows 5:30 --years_out 2005-2006 --stages c4 >> out_nsh.log 2>&1
$CMP out_new out_nsh --rows 5:30 --years 2005-2006 && ok "n4: default, E5-style sharded chain (--lat_rows 5:30 --years_out) == rows+slice of default" || bad "n4"

# (n5) hybrid interim: older years carry a spatial_ref coordinate (reference odc files), newer do not
uv run -q --with xarray --with netCDF4 python $S/make_mixed.py $S
rm -rf out_mixed; $STEP2 --interim_path $S/data_mixed/interim --static_path $S/data/static --results_path out_mixed --lat_rows 5:30 --years_out 2005-2006 > out_mixed.log 2>&1 || tail -5 out_mixed.log
$CMP out_new out_mixed --rows 5:30 --years 2005-2006 && ok "n5: mixed interim (spatial_ref in some years) == default" || bad "n5: mixed interim"

# (h) E10: fatal --lat_rows mismatch, vegfrac skip via _exists, ΔT time values, dead worker
# (h1) full-grid run into a dir holding --lat_rows 10:25 outputs -> exit 1
rm -rf out_h1; mkdir out_h1; cp out_r/LAI_500m.nc out_h1/
$STEP2 $BASE --results_path out_h1 --stages gapfill --vars LAI > out_h1.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "ERROR: .*lat_rows=10:25" out_h1.log && ok "h1: lat_rows mismatch of an existing output is fatal (rc=$rc)" || bad "h1 rc=$rc"
# (h2) vegfrac: only trees exists, from another window -> fatal (was: silently recomputed)
rm -rf out_h2; mkdir out_h2; cp out_r/trees_500m.nc out_h2/
$STEP2 $BASE --results_path out_h2 --stages vegfrac > out_h2.log 2>&1; rc=$?
[ $rc = 1 ] && grep -q "ERROR: .*trees_500m.nc" out_h2.log && ok "h2: vegfrac skip check goes through _exists (rc=$rc)" || bad "h2 rc=$rc"
# (h2b) vegfrac all three present for the same window -> skip
rm -rf out_h2b; mkdir out_h2b; cp out_r/{trees,grass,bare}_500m.nc out_h2b/
$STEP2 $BASE --results_path out_h2b --lat_rows 10:25 --stages vegfrac > out_h2b.log 2>&1; rc=$?
[ $rc = 0 ] && grep -q "trees/grass/bare: exist, skipping" out_h2b.log && ok "h2b: vegfrac same-window skip" || bad "h2b rc=$rc"
# (h3) ΔT: Tavg time values shifted by one day (same length) -> error
rm -rf data_dt; mkdir -p data_dt/interim
for d in data/interim/*; do v=$(basename $d); [ $v = Tavg ] || ln -s $S/$d data_dt/interim/$v; done
mkdir data_dt/interim/Tavg
uv run -q --with xarray --with netCDF4 python - <<PY
import xarray as xr, glob, os, pandas as pd
for f in sorted(glob.glob('$S/data/interim/Tavg/*.nc')):
    ds = xr.load_dataset(f)
    ds = ds.assign_coords(time=pd.DatetimeIndex(ds.time.values) + pd.Timedelta(days=1))
    ds.to_netcdf('$S/data_dt/interim/Tavg/' + os.path.basename(f))
PY
rm -rf out_h3
$STEP2 --interim_path $S/data_dt/interim --static_path $S/data/static --results_path out_h3 --stages deltat > out_h3.log 2>&1; rc=$?
[ $rc != 0 ] && grep -q "first mismatch at index 0" out_h3.log && [ ! -e out_h3/ΔT_500m.nc ] && ok "h3: ΔT LST/Tavg time values compared (rc=$rc)" || bad "h3 rc=$rc"
# (h4) a killed worker -> error, not a hang
rm -rf out_h4 out_h4.rc
( $STEP2 $BASE --results_path out_h4 --stages gapfill --vars NDWI --threads 3 --tile_rows 1 > out_h4.log 2>&1; echo $? > out_h4.rc ) &
bg=$!
victim=""
# victim = a forked worker: a matching process whose grandparent also matches (uv -> python -> worker)
for i in $(seq 1 600); do
  m=" $(pgrep -d ' ' -f "results_path out_h4") "
  victim=""
  for p in $m; do
    pp=$(ps -o ppid= -p $p | tr -d ' '); [ -n "$pp" ] || continue
    gp=$(ps -o ppid= -p $pp | tr -d ' ')
    case "$m" in *" $pp "*) case "$m" in *" $gp "*) victim=$p;; esac;; esac
  done
  [ -n "$victim" ] && break
  sleep 0.1
done
kill -9 $victim 2>/dev/null
for i in $(seq 1 600); do [ -s out_h4.rc ] && break; sleep 0.2; done
rc=$(cat out_h4.rc 2>/dev/null || echo hang)
[ "$rc" != hang ] && [ "$rc" != 0 ] && grep -q "worker process died" out_h4.log && [ ! -e out_h4/NDWI_500m.nc ] && ! ls out_h4/*.part >/dev/null 2>&1 && ok "h4: killed worker -> BrokenProcessPool error (rc=$rc, no output/.part left)" || { bad "h4 rc=$rc"; pkill -9 -f "results_path out_h4"; }
echo; echo "PASSED: ${#PASS[@]}"; printf '  %s\n' "${PASS[@]}"
echo "FAILED: ${#FAIL[@]}"; printf '  %s\n' "${FAIL[@]}"
