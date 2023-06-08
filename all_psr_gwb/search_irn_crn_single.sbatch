#!/bin/bash

#SBATCH --job-name=gwb_search
#SBATCH --output=search_irn_crn.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

python search_irn_crn.py -c chains_single

echo "Run complete."
