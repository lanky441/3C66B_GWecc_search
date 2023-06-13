#!/bin/bash

#SBATCH --job-name=psrterm_4mpi

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=4

which python
bash print_info.sh

mpirun -np 4 python search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm.json

echo "Run complete."
