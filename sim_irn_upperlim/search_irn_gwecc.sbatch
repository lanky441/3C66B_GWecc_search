#!/bin/bash

#SBATCH --job-name=search_irn_gwecc
#SBATCH --output=search_irn_gwecc.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

which python

mpirun -np 4 python search_irn_gwecc.py

echo "Run complete."
