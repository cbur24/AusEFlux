#!/bin/bash
 
#PBS -P er8
#PBS -q normalsr
#PBS -l ncpus=52
#PBS -l mem=400gb
#PBS -l walltime=06:00:00
#PBS -l jobfs=1500MB
#PBS -l storage=gdata/os22+gdata/xc0+scratch/xc0
 
module load python3/3.10.0
source /g/data/xc0/project/AusEFlux/env/py310/bin/activate
# source /g/data/os22/chad_tmp/AusENDVI/env/py310/bin/activate

python3 /g/data/xc0/project/AusEFlux/src/_batch_run_ensemble.py