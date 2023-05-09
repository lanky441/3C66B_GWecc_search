#!/bin/bash

#SBATCH --job-name=search_crn_gwecc2
#SBATCH --output=search_crn_gwecc2.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

which python

mpirun -np 4 python search_crn_gwecc.py

echo "Run complete."
