#!/bin/bash

#SBATCH --job-name=search_crn_gwecc_single
#SBATCH --output=search_crn_gwecc_single.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

python search_crn_gwecc.py

echo "Run complete."
