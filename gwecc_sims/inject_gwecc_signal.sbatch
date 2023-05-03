#!/bin/bash

#SBATCH --job-name=injection_gwecc
#SBATCH --output=inject_gwecc_signal.out
#SBATCH -p mamclaughlin
#
#SBATCH --ntasks=1
#SBATCH --mem=64G

which python

for n in {0..2}; do python inject_gwecc_signal.py -n $n; done

echo "All pulsar injection donw."
