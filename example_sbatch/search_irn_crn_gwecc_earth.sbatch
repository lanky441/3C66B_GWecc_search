#!/bin/bash

#SBATCH --job-name=earth_2_temp
#SBATCH --output=search_irn_crn_gwecc_earth.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=2

module load parallel/openmpi/3.1.6_gcc93

which python
which mpirun
which mpicc

mpirun -np 2 python search_irn_crn_gwecc_earth.py -hc -n 2000000

echo "Run complete."
