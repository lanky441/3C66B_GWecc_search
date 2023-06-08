#!/bin/bash

#SBATCH --job-name=ten_5f_E_1_temp
#SBATCH --output=search_irn_crn_gwecc_psrterm_single_t2equad_10psr.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1

which python

python search_irn_crn_gwecc_psrterm.py -c chains_single_t2equad_10psr

echo "Run complete."
