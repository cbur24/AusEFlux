#!/bin/bash
 
#PBS -P u46
#PBS -q normalsr
#PBS -l ncpus=52
#PBS -l mem=240gb
#PBS -l walltime=03:00:00
#PBS -l storage=gdata/os22+gdata/xc0
 
module load python3/3.10.0
source /g/data/xc0/project/AusEFlux/env/py310/bin/activate

python3 /g/data/xc0/project/AusEFlux/src/_batch_run_ensemble.py