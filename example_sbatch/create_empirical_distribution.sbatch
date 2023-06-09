#!/bin/bash

#SBATCH --job-name=emp_dist
#SBATCH --output=create_empirical_distribution_5f.out
#SBATCH -p standby
#
#SBATCH --ntasks=1

which python

python create_empirical_distribution.py

echo "Run complete."
