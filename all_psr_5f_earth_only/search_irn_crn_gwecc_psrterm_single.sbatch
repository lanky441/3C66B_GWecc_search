#!/bin/bash

#SBATCH --job-name=all_1_temp
#SBATCH --output=search_irn_crn_gwecc_psrterm_single.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

python search_irn_crn_gwecc_psrterm.py -c chains_single

echo "Run complete."
