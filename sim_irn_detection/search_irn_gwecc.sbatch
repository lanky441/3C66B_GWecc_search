#!/bin/bash

#SBATCH --job-name=search_irn_gwecc_true
#SBATCH --output=search_irn_gwecc_true.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

which python

mpirun -np 4 python search_irn_gwecc.py

echo "Run complete."
