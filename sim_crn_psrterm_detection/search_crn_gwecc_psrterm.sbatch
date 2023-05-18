#!/bin/bash

#SBATCH --job-name=search_crn_gwecc_psrterm
#SBATCH --output=search_crn_gwecc_psrterm_fixedpdist.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

which python

mpirun -np 4 python search_crn_gwecc_psrterm.py

echo "Run complete."
