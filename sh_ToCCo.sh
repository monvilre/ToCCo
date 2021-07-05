#!/bin/bash

#OAR -n ToCCo

#OAR -l /nodes=1/core=4,walltime=04:00:00

# Modules loading

source /soft/env.bash
module load python/python3.7

# Launch compute job

python3 /nfs_scratch/$USER/tocco/script_pop.py
