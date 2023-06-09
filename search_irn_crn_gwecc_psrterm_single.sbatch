#!/bin/bash

#SBATCH --job-name=psrterm_all
#SBATCH --output=search_irn_crn_gwecc_psrterm_single.out
#SBATCH -p mamclaughlin
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

which python

python search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm.json

echo "Run complete."
