#!/bin/bash

#SBATCH --job-name=det_4_temp
#SBATCH --output=search_crn_gwecc_psrterm.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

module load parallel/openmpi/3.1.6_gcc93

which python
which mpirun
which mpicc

mpirun -np 4 python search_crn_gwecc_psrterm.py -hc
echo "Run complete."
