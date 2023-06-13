#!/bin/bash

#SBATCH --job-name=psrterm_2_psr

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=4

#module load parallel/openmpi/3.1.6_gcc93
#export MPICC=`which mpicc`

which python
#bash print_info.sh

mpirun -np 4 search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm_J1909.json

echo "Run complete."
