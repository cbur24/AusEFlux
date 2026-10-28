#!/bin/bash
# check_inputs.sh YEAR : list the raw inputs step1 needs for YEAR and say
# which are missing (exit 1 if any).  Also checks the static rasters, the
# models, the variables file, the envs and the history interim coverage.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh" "$1"
y=$YEAR; yp=$((YEAR - 1)); miss=0
chk() { if [[ -e "$1" ]]; then printf '  ok       %s\n' "$1"; else printf '  MISSING  %s\n' "$1"; miss=$((miss + 1)); fi; }
echo "Raw inputs for $y under $RAW:"
for f in MODIS/mosaic/MCD43A4.006/MCD43A4.006.b01.500m_0620_0670nm_nbar.$y.nc \
         MODIS/mosaic/MCD43A4.006/MCD43A4.006.b02.500m_0841_0876nm_nbar.$y.nc \
         MODIS/mosaic/MCD43A4.006/MCD43A4.006.b05.500m_1230_1250nm_nbar.$y.nc \
         MODIS/mosaic/MYD11A1.006/MYD11A1.006.LST_Day_1km.$y.nc \
         MODIS/mosaic/MYD11A1.006/MYD11A1.006.QC_Day.$y.nc \
         OzWALD/8day/NDVI/OzWALD.NDVI.$y.nc OzWALD/8day/LAI/OzWALD.LAI.$y.nc \
         OzWALD/daily/meteo/Tmin/OzWALD.Tmin.$y.nc OzWALD/daily/meteo/Tmax/OzWALD.Tmax.$y.nc \
         OzWALD/daily/meteo/kTavg/OzWALD.kTavg.$y.nc \
         SILO/radiation/$y.radiation.nc SILO/daily_rain/$y.daily_rain.nc SILO/vp/$y.vp.nc \
         SILO/daily_rain/$yp.daily_rain.nc; do chk "$RAW/$f"; done
echo "Completeness of the $y files (time steps; a full year is 365/366 daily, 46 8-day, 12 monthly):"
if [[ -e $RAW/SILO/daily_rain/$y.daily_rain.nc ]] && command -v python >/dev/null; then
  python - "$RAW" "$y" <<'PY' 2>/dev/null || echo "  (python/netCDF4 not available on this node; activate $WORK/envs/activate_step12.sh)"
import sys, netCDF4, glob
raw, y = sys.argv[1], sys.argv[2]
for pat in [f'{raw}/MODIS/mosaic/MCD43A4.006/*b02*{y}.nc', f'{raw}/MODIS/mosaic/MYD11A1.006/*LST_Day*{y}.nc',
            f'{raw}/OzWALD/8day/NDVI/*{y}.nc', f'{raw}/OzWALD/daily/meteo/Tmax/*{y}.nc', f'{raw}/SILO/daily_rain/{y}*.nc']:
    for f in glob.glob(pat):
        with netCDF4.Dataset(f) as d:
            t = d['time']; n = len(t)
            print(f'  {n:4d} steps  {f}  (last: {netCDF4.num2date(t[-1], t.units, getattr(t, "calendar", "standard"))})')
PY
fi
echo "Static rasters, models, variables, envs:"
for f in $STATIC/land_sea_mask_500m.nc $STATIC/urban_water_mask_500m.nc $STATIC/ndvi_of_baresoil_500m.nc \
         $STATIC/VegH_1km_2007_2010.nc $STATIC/Aust_C4_grass_cover_percentage.tif $STATIC/grid_500m \
         $VARIABLES_FILE $WORK/envs/activate_step12.sh $WORK/envs/activate_step3.sh; do chk "$f"; done
for F in $FLUXES; do chk "$MODELS/$F"; [[ -d $MODELS_COMPILED/$F/compiled ]] && echo "  ok       compiled $F ($(ls $MODELS_COMPILED/$F/compiled | wc -l) .so)" || echo "  todo     compile $F (run_year.sh does it)"; done
echo "History interim (2003..$yp, rain from 2002) found in INTERIM_SOURCES=$INTERIM_SOURCES:"
for v in $INTERIM_VARS; do
  y0=2003; [[ $v == rain ]] && y0=2002; gap=""
  for yy in $(seq $y0 $yp); do
    found=0; for src in $(IFS=:; for s in $INTERIM_SOURCES; do eval echo $s; done); do [[ -e $src/$v/${v}_500m_$yy.nc ]] && { found=1; break; }; done
    [[ $found == 0 ]] && gap="$gap $yy"
  done
  if [[ -n $gap ]]; then echo "  MISSING  $v:$gap"; miss=$((miss + 1)); else echo "  ok       $v $y0-$yp"; fi
done
[[ -n ${CLIM_SRC:-} ]] && { [[ -d $CLIM_SRC ]] && echo "Stored climatologies: $(ls $CLIM_SRC | wc -l) files in $CLIM_SRC" || { echo "MISSING climatology dir $CLIM_SRC (set CLIM_SRC= to compute fresh, +100 SU)"; miss=$((miss + 1)); }; }
echo; [[ $miss == 0 ]] && echo "ALL INPUTS PRESENT for $y" || echo "$miss item(s) missing for $y"
exit $(( miss > 0 ))
