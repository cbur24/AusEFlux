#!/bin/bash
# Smoke runs of step3_ensemble_prediction.py / _quantile.py on the synthetic tree.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
export AUSEFLUX_ROOT=$S/root
rm -rf $S/out $S/dumps $S/root_inf $S/root_win $S/root_win_noattr $S/root_tbad
mkdir -p $S/out
# expect_rc NAME CMD...: run CMD, record its exit status in out/NAME.status (log in out/NAME.log)
expect_rc() { local n=$1; shift; set +e; "$@" > $S/out/$n.log 2>&1; echo $? > $S/out/$n.status; set -e; }
# The matrix below was written for the pre-2026-10-27 defaults (mask without
# LAI_anom, +-Inf counted as no-data); those are now explicit (OLD).  The new
# defaults (nearest, LAI_anom mask, --mask_test nan, valid_only) are checked
# in the "defaults" block at the end.
OLD="--mask_vars VegH,NDWI,rain_anom,Tavg --mask_test nonfinite"
E0="uv run --quiet $REPO/step3_ensemble_prediction.py --year 2024 --fluxes GPP"
Q0="uv run --quiet $REPO/step3_ensemble_prediction_quantile.py --year 2024 --fluxes GPP"
E="$E0 $OLD"
Q="$Q0 $OLD"
for m in linear nearest; do
  $E --quantile_method $m --all_pixels --output_path $S/out/${m}_all  --dump_members $S/dumps/${m}_all
  $E --quantile_method $m --valid_only --output_path $S/out/${m}_valid --dump_members $S/dumps/${m}_valid
  # parts of one product must share the configuration attributes (incl. valid_only):
  # linear shards all_pixels, nearest shards valid_only
  PX=$([ $m = linear ] && echo --all_pixels || echo --valid_only)
  $E --quantile_method $m $PX --months 1-6  --output_path $S/out/${m}_shard --dump_members $S/dumps/${m}_shard
  $E --quantile_method $m $PX --months 7-12 --output_path $S/out/${m}_shard --dump_members $S/dumps/${m}_shard
  $E --quantile_method $m --merge --output_path $S/out/${m}_shard
done
# reference mask list + first 3 members (option plumbing only)
$E0 --quantile_method nearest --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --n_models 3 --threads 2 --output_path $S/out/refmask_n3
# small tiles (TILE_ROWS=7) to exercise tile boundaries in valid_only
uv run --quiet $S/tiles_wrapper.py --year 2024 --fluxes GPP $OLD --quantile_method nearest --valid_only --output_path $S/out/nearest_tiles7
# quantile (fast) path
$Q --all_pixels --output_path $S/out/q_all
$Q --valid_only --output_path $S/out/q_valid
$Q --months 1-6 --valid_only --output_path $S/out/q_shard
$Q --months 7-12 --valid_only --output_path $S/out/q_shard
$Q --merge --output_path $S/out/q_shard
# --- latitude window (--lat_rows 10:25) ---
W="--lat_rows 10:25"
for m in linear nearest; do
  $E --quantile_method $m --all_pixels $W --output_path $S/out/${m}_all --dump_members $S/dumps/${m}_all
  $E --quantile_method $m --valid_only $W --output_path $S/out/${m}_valid --dump_members $S/dumps/${m}_valid
  PX=$([ $m = linear ] && echo --all_pixels || echo --valid_only)
  $E --quantile_method $m $PX --months 1-6  $W --output_path $S/out/${m}_wshard --dump_members $S/dumps/${m}_wshard
  $E --quantile_method $m $PX --months 7-12 $W --output_path $S/out/${m}_wshard --dump_members $S/dumps/${m}_wshard
  $E --quantile_method $m --merge $W --output_path $S/out/${m}_wshard
done
# merging windowed parts into a full product must be refused
if $E --merge --output_path $S/out/nearest_wshard > $S/out/merge_refuse.log 2>&1; then
  echo "refuse-merge: UNEXPECTED SUCCESS" > $S/out/merge_refuse.status
else
  echo "refuse-merge: exit $?" > $S/out/merge_refuse.status
fi
# windowed + TILE_ROWS=7 (tiles start at START: 10-17,17-24,24-25)
uv run --quiet $S/tiles_wrapper.py --year 2024 --fluxes GPP $OLD --quantile_method nearest --valid_only $W --output_path $S/out/nearest_tiles7
$Q --all_pixels $W --output_path $S/out/q_all
$Q --valid_only $W --output_path $S/out/q_valid
$Q --months 1-6 --valid_only $W --output_path $S/out/q_wshard
$Q --months 7-12 --valid_only $W --output_path $S/out/q_wshard
$Q --merge $W --output_path $S/out/q_wshard
if $Q --merge --output_path $S/out/q_wshard > $S/out/q_merge_refuse.log 2>&1; then
  echo "refuse-merge: UNEXPECTED SUCCESS" > $S/out/q_merge_refuse.status
else
  echo "refuse-merge: exit $?" > $S/out/q_merge_refuse.status
fi

# --- new defaults (2026-10-27) ---
REF="--quantile_method nearest --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --mask_test nan"
$E0 --output_path $S/out/def_default
$E0 $REF --valid_only --output_path $S/out/def_explicit
$E0 $REF --all_pixels --output_path $S/out/def_explicit_all
# old defaults made explicit; compared with saved old-code outputs (out_prev/linear_all)
$E0 --quantile_method linear --mask_vars VegH,NDWI,rain_anom,Tavg --mask_test nonfinite --all_pixels --output_path $S/out/def_old
$Q0 --output_path $S/out/q_def_default
$Q0 --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --mask_test nan --valid_only --output_path $S/out/q_def_explicit
$Q0 --mask_vars LAI_anom,VegH,NDWI,rain_anom,Tavg --mask_test nan --all_pixels --output_path $S/out/q_def_explicit_all
$Q0 --mask_vars VegH,NDWI,rain_anom,Tavg --mask_test nonfinite --all_pixels --output_path $S/out/q_def_old
# +-Inf in mask variables (LAI_anom, NDWI): nan vs nonfinite must differ, and
# valid_only == all_pixels must still hold under --mask_test nan
uv run --quiet --with xarray --with netcdf4 python $S/make_inf_features.py $S/root/features_500m $S/root_inf/features_500m
F="--features_path $S/root_inf/features_500m"
$E0 $F --output_path $S/out/inf_default
$E0 $F --all_pixels --output_path $S/out/inf_all
$E0 $F --mask_test nonfinite --output_path $S/out/inf_nonfinite
$Q0 $F --output_path $S/out/q_inf_default
$Q0 $F --all_pixels --output_path $S/out/q_inf_all
$Q0 $F --mask_test nonfinite --output_path $S/out/q_inf_nonfinite

# --- review round 2 (2026-10-27) ---
# (g) latitude-windowed FEATURES (auseflux_lat_rows=10:25) without --lat_rows:
#     outputs named _r10-25, urban mask sliced to rows 10:25
uv run --quiet --with xarray --with netcdf4 --with pandas python $S/make_variant_features.py $S/root/features_500m $S/root_win/features_500m win
uv run --quiet --with xarray --with netcdf4 --with pandas python $S/make_variant_features.py $S/root/features_500m $S/root_win_noattr/features_500m win_noattr
uv run --quiet --with xarray --with netcdf4 --with pandas python $S/make_variant_features.py $S/root/features_500m $S/root_tbad/features_500m tbad
FW="--features_path $S/root_win/features_500m"
expect_rc winfeat $E0 $FW --output_path $S/out/winfeat
expect_rc winfeat_all $E0 $FW --all_pixels --output_path $S/out/winfeat_all
expect_rc q_winfeat $Q0 $FW --output_path $S/out/q_winfeat
#     --lat_rows (full-grid rows) inside the feature window
expect_rc winfeat_sub $E0 $FW --lat_rows 12:20 --output_path $S/out/winfeat_sub
expect_rc q_winfeat_sub $Q0 $FW --lat_rows 12:20 --output_path $S/out/q_winfeat_sub
#     --lat_rows outside the feature window -> error
expect_rc winfeat_outside $E0 $FW --lat_rows 5:20 --output_path $S/out/winfeat_outside
expect_rc q_winfeat_outside $Q0 $FW --lat_rows 5:20 --output_path $S/out/q_winfeat_outside
#     windowed features WITHOUT the attribute -> urban mask mismatch error
expect_rc winfeat_noattr $E0 --features_path $S/root_win_noattr/features_500m --output_path $S/out/winfeat_noattr
expect_rc q_winfeat_noattr $Q0 --features_path $S/root_win_noattr/features_500m --output_path $S/out/q_winfeat_noattr
# (h) time-axis mismatch between features -> error naming the feature
expect_rc tbad $E0 --features_path $S/root_tbad/features_500m --output_path $S/out/tbad
expect_rc q_tbad $Q0 --features_path $S/root_tbad/features_500m --output_path $S/out/q_tbad
# (i) nothing produced -> non-zero; already exists -> 0 (+ WARNING on config mismatch)
mkdir -p $S/out/empty_models/GPP
expect_rc nothing $E0 --models_path $S/out/empty_models --output_path $S/out/nothing
expect_rc nothing_nofolder $E0 --models_path $S/out/no_such_models --output_path $S/out/nothing
expect_rc q_nothing $Q0 --models_path $S/out/no_such_models --output_path $S/out/q_nothing
expect_rc nomonth $E0 --months 1-12 --features_path $S/root_win/features_500m --output_path $S/out/nomonth_dummy --year 2023
expect_rc exists_ok $E0 --output_path $S/out/def_default
expect_rc exists_warn $E0 --mask_test nonfinite --output_path $S/out/def_default
expect_rc q_exists_warn $Q0 --all_pixels --output_path $S/out/q_def_default
expect_rc exists_dumps $E0 --output_path $S/out/def_default --dump_members $S/dumps/none_yet
# (j) --merge_partial and merge refusal on configuration mismatch
$E0 --months 1-3 --output_path $S/out/partial
$Q0 --months 1-3 --output_path $S/out/q_partial
expect_rc partial_strict $E0 --merge --output_path $S/out/partial
expect_rc partial $E0 --merge_partial --output_path $S/out/partial
expect_rc q_partial $Q0 --merge_partial --output_path $S/out/q_partial
$E0 --months 1-6 --valid_only --output_path $S/out/mix
$E0 --months 7-12 --all_pixels --output_path $S/out/mix
expect_rc mix_merge $E0 --merge --output_path $S/out/mix
$Q0 --months 1-6 --output_path $S/out/q_mix
$Q0 --months 7-12 --mask_test nonfinite --output_path $S/out/q_mix
expect_rc q_mix_merge $Q0 --merge --output_path $S/out/q_mix
# (k) argument validation
expect_rc bad_nmodels $E0 --n_models 0 --output_path $S/out/badargs
expect_rc bad_threads $E0 --threads 0 --output_path $S/out/badargs
expect_rc q_bad_threads $Q0 --threads 0 --output_path $S/out/badargs
expect_rc bad_months $E0 --months 1-13 --output_path $S/out/badargs
