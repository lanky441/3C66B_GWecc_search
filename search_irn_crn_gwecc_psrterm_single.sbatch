#!/bin/bash

#SBATCH --job-name=psrterm_all_psr

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=1

which python
bash print_info.sh

python search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm_J1909.json

echo "Run complete."
