#!/bin/bash

#SBATCH --job-name=earth_all_psr

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=1

which python
bash print_info.sh

python search_irn_crn_gwecc.py -s irn_crn_gwecc_earth.json

echo "Run complete."
