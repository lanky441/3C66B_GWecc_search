#!/bin/bash

#SBATCH --job-name=real_gwecc_single
#SBATCH --output=search_irn_crn_gwecc_single.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

# mpirun -np 4 python search_irn_crn_gwecc.py -hc
python search_irn_crn_gwecc.py -c chains_single

echo "Run complete."
