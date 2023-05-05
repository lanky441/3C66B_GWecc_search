#!/bin/bash

#SBATCH -J search_crn_gwecc
#SBATCH -o search_crn_gwecc.out
#SBATCH -p standby
#SBATCH --mem=64G
#SBATCH -n 4

which python

mpirun -np 4 python search_crn_gwecc.py

echo "Run complete."
