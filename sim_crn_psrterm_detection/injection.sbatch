#!/bin/bash

#SBATCH --job-name=inject_irn
#SBATCH --output=inject_irn.out
#SBATCH -p standby
#
#SBATCH --ntasks=1
#SBATCH --mem=64G

which python

psrlist="J1909-3744 J2317+1439 J2043+1711 J1600-3053 J1918-0642 J0613-0200 J1944+0907 J1744-1134 J1910+1256 J0030+0451"

for psr in $psrlist; do python injection.py -n $psr -irn; done

echo "All pulsar injection done."
