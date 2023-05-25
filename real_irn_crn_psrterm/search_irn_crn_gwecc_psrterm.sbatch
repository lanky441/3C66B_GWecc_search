#!/bin/bash

#SBATCH --job-name=gwecc_psrterm_emp
#SBATCH --output=search_irn_crn_gwecc_psrterm_empdist.out
#SBATCH -p standby
#
#SBATCH --ntasks=4

module load parallel/openmpi/3.1.6_gcc93
export MPICC=`which mpicc`

which python
which mpirun
which mpicc

mpirun -np 4 python search_irn_crn_gwecc_psrterm.py -hc -c chains_4_emp/
#python search_irn_crn_gwecc_psrterm.py -c chains_single_emp

echo "Run complete."
