#!/bin/bash

#SBATCH -J search_crn
#SBATCH -o earch_crn.out
#SBATCH -p mamclaughlin
#
#SBATCH -n 1

which python

python search_crn.py

echo "Run complete."
