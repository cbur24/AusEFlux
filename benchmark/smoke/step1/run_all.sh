#!/bin/bash
# step1 smoke battery: $1 = odc-geo requirement (e.g. odc-geo or odc-geo==0.4.2), $2 = out prefix
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}; S=$REPO/step1_harmonisation.py; O=$1; P=$2; mkdir -p $P
UVP="uv run --with xarray --with netcdf4 python"
rm -rf $P/*
uv run --script $S --year 2024 --input_path $PWD/data/input --results_path $PWD/$P/default > $P/log_default.txt 2>&1 || echo FAIL default
uv run --script $S --year 2024 --input_path $PWD/data/input --results_path $PWD/$P/newqc --lst_qc new > $P/log_newqc.txt 2>&1 || echo FAIL newqc
uv run --with "$O" --script $S --year 2024 --input_path $PWD/data/input --results_path $PWD/$P/odc --engine odc > $P/log_odc.txt 2>&1 || echo FAIL odc
uv run --script $S --year 2024 --input_path $PWD/data/input --results_path $PWD/$P/poison --modis_nodata poison > $P/log_poison.txt 2>&1 || echo FAIL poison
for eng in rasterio odc; do for m in exclude poison; do
 uv run --with "$O" --script $S --year 2024 --input_path $PWD/data_nan/input --static_path $PWD/data/input/static --results_path $PWD/$P/nan_${eng}_$m --vars NDWI,kNDVI --engine $eng --modis_nodata $m > $P/log_nan_${eng}_$m.txt 2>&1 || echo FAIL nan $eng $m
done; done
echo "== verify default vs previous out_default (expect all 0 except LST: default --lst_qc is now ref)"; $UVP verify.py $P/default out_default 2>&1 | grep -v Installed
BC="uv run --quiet --with numpy --with netcdf4 python bitcmp.py"
echo "== bitcmp --lst_qc new vs saved out_default (pre-flip default; expect BIT-IDENTICAL, 14 files)"; $BC $P/newqc out_default
echo "== bitcmp new default (lst_qc ref) vs saved out_default, all but LST (expect BIT-IDENTICAL, 13 files)"; $BC $P/default out_default NDWI kNDVI NDVI LAI VegH Tmin Tmax kTavg Tavg SRAD rain VPD
echo "== bitcmp new default LST vs saved out_qcref (expect BIT-IDENTICAL, 1 file)"; $BC $P/default out_qcref LST
echo "== no .part files left"; find $P -name '*.part.*' | wc -l
echo "== atomic/race checks"; uv run --quiet --with numpy --with xarray --with netcdf4 --with pandas --with rasterio --with rioxarray --with dask python atomic_check.py $PWD $P/atomic 2>&1 | grep -v Installed
echo "== verify odc vs rasterio"; $UVP verify.py $P/odc $P/default 2>&1 | grep -v Installed
echo "== verify poison (aligned smoke grid) vs default"; $UVP verify.py $P/poison $P/default 2>&1 | grep -v Installed
echo "== poison check (offset grid with NaN)"; (cd $P && ln -sfn . x && $UVP ../poison_check.py 2>&1 | grep -v Installed)
echo "== wide grid"; uv run --with "$O" --with rasterio --with rioxarray --with xarray --with dask --with netcdf4 --with pandas python wide_grid_check.py $S 2>&1 | grep -v -e Installed -e SyntaxWarning -e kwargs
