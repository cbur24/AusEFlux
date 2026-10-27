#!/bin/bash
# cmp.sh NAME NEWFILE FLUX YEAR [ncpus mem walltime]
E=/scratch/ub8/pl5189/auseflux/results/e5
P=/scratch/ub8/pl5189/auseflux/code/AusEFlux/benchmark/gadi/pbs/compare.pbs
name=$1 new=$2 flux=$3 yr=$4
res=${5:-ncpus=4,mem=16GB,walltime=01:00:00}
ref=/g/data/xc0/project/AusEFlux/results/AusEFlux/v2.1/$flux/AusEFlux_${flux}_500m_quantiles_${yr}_v2.1.nc
ex="--mvg /g/data/xc0/project/AusEFlux/data/aus6_0e_mvg.tif --plots"
[ $flux = NEE ] && ex="$ex --thresholds $E/th_nee.json"
qsub -N c_$name -o $E/logs/cmp_$name.log -l $res ${DEP:+-W depend=afterok:$DEP} \
  -v AUSEF_ENV=step12,NEW=$new,REF=$ref,FLUX=$flux,OUT=$E/cmp/$name,EXTRA="$ex" $P
