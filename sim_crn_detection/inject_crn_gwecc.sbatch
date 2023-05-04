#!/bin/bash

#SBATCH --job-name=inject_crn_gwecc
#SBATCH --output=inject_crn_gwecc.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1
#SBATCH --mem=64G

which python

for n in {4,6,16,23,28,29,31,33,38,45}; do python inject_crn_gwecc.py -n $n; done

echo "All pulsar injection done."
