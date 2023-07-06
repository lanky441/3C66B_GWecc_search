#!/bin/bash

#SBATCH --job-name=test

#SBATCH -p standby
#SBATCH --mem=64G
#SBATCH --ntasks=1

which python
#bash print_info.sh

python search_irn_crn_gwecc_fixed_eta.py -s irn_crn_gwecc_psrterm_J1909.json

echo "Run complete."
