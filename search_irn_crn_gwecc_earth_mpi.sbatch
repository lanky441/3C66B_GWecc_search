#!/bin/bash

#SBATCH --job-name=ear_4_23Jun

#SBATCH -p mamclaughlin
#SBATCH --mem=128G
#SBATCH --ntasks=4

which python
bash print_info.sh

mpirun -np 4 python search_irn_crn_gwecc.py -s irn_crn_gwecc_earth.json

echo "Run complete."
