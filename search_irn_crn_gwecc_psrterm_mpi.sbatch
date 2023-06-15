#!/bin/bash

#SBATCH --job-name=psrterm_2mpi

#SBATCH -p mamclaughlin
#SBATCH --mem=64G
#SBATCH --ntasks=2

which python
bash print_info.sh

mpirun -np 2 python search_irn_crn_gwecc.py -s irn_crn_gwecc_psrterm.json

echo "Run complete."
