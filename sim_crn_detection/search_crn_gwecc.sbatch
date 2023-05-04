#!/bin/bash

#SBATCH --job-name=search_crn_gwecc
#SBATCH --output=search_crn_gwecc.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=8
#SBATCH --mem=64G

which python

mpirun -np 4 python search_crn_gwecc.py

echo "Run complete."
