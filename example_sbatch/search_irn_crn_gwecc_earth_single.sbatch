#!/bin/bash

#SBATCH --job-name=earth_vary_DE
#SBATCH --output=search_irn_crn_gwecc_earth_single.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

python search_irn_crn_gwecc_earth.py -c chains_single_vary_DE -n 2000000

echo "Run complete."
