#!/bin/bash

#SBATCH --job-name=real_gwecc_psrterm
#SBATCH --output=search_irn_crn_gwecc_psrterm.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=4

which python

mpirun -np 4 python search_irn_crn_gwecc_psrterm.py -hc
# python search_irn_crn_gwecc_psrterm.py -c chains_single

echo "Run complete."
