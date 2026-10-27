#!/bin/bash
# continental 2025 chain + 2024 windows
set -eu
E=/scratch/ub8/pl5189/auseflux/results/e5
P=$E/e5_step2.pbs
F=/scratch/ub8/pl5189/auseflux/features_500m_e5
L=$E/logs
R="ncpus=8,mem=32GB,walltime=06:00:00"
q() { local name=$1; shift; qsub -N $name -o $L/$name.log -l $R "$@" $P; }
declare -A G
for grp in NDWI:kNDVI NDVI:LAI LST:Tavg SRAD:VPD rain:VegH; do
  n=gf_${grp//:/_}; G[$grp]=$(q $n -v OUTDIR=$F,YEARS=2025-2025,STAGE=gapfill,VARS=$grp); echo "$n ${G[$grp]}"
done
VF=$(q vegfrac -W depend=afterok:${G[NDVI:LAI]} -v OUTDIR=$F,YEARS=2025-2025,STAGE=vegfrac); echo "vegfrac $VF"
CR=$(q cmlrain -W depend=afterok:${G[rain:VegH]} -v OUTDIR=$F,YEARS=2025-2025,STAGE=cmlrain); echo "cmlrain $CR"
DT=$(q deltat -W depend=afterok:${G[LST:Tavg]}:${G[rain:VegH]}:${G[NDVI:LAI]}:${G[SRAD:VPD]}:${G[NDWI:kNDVI]} -v OUTDIR=$F,YEARS=2025-2025,STAGE=deltat); echo "deltat $DT"
ALLG=${G[LST:Tavg]}:${G[rain:VegH]}:${G[NDVI:LAI]}:${G[SRAD:VPD]}:${G[NDWI:kNDVI]}
for v in NDWI SRAD Tavg VPD kNDVI LAI rain rain_cml3 rain_cml6 rain_cml12; do
  id=$(q an_$v -W depend=afterok:$ALLG:$CR -v OUTDIR=$F,YEARS=2025-2025,STAGE=anom,VARS=$v); echo "an_$v $id"
done
C4=$(q c4 -W depend=afterok:$VF -v OUTDIR=$F,YEARS=2025-2025,STAGE=c4); echo "c4 $C4"
