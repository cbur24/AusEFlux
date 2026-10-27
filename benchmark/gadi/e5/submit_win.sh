#!/bin/bash
set -eu
E=/scratch/ub8/pl5189/auseflux/results/e5
for w in north:600:1000 centre:2800:3200 south:5000:5400; do
  n=${w%%:*}; rows=${w#*:}
  id=$(qsub -N w24_$n -o $E/logs/w24_$n.log -l ncpus=8,mem=32GB,walltime=02:00:00 -v OUTDIR=$E/features_w$n,YEARS=2024-2024,ROWS=$rows $E/e5_step2.pbs); echo "w24_$n $id"
done
