#!/bin/bash

#SBATCH --job-name=psr_1_A_nogrp

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=1

which python
bash print_info.sh

python search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm.json

echo "Run complete."
