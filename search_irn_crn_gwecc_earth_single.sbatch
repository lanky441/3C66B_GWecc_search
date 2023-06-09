#!/bin/bash

#SBATCH --job-name=earth_J1909
#SBATCH --output=search_irn_crn_gwecc_earth_single.out
#SBATCH -p mamclaughlin
#SBATCH --mem-per-cpu=64G
#SBATCH --ntasks=1

which python

python search_irn_crn_gwecc.py -s irn_crn_gwecc_earth.json

echo "Run complete."
