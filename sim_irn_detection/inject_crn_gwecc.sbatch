#!/bin/bash

#SBATCH --job-name=inject_crn_gwecc
#SBATCH --output=inject_crn_gwecc.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1
#SBATCH --mem=64G

which python

for n in {0..2}; do python inject_crn_gwecc.py -n $n; done

echo "All pulsar injection donw."
