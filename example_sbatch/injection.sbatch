#!/bin/bash

#SBATCH --job-name=inject_crn_gwecc_psrterm
#SBATCH --output=inject_crn_gwecc_psrterm.out
#SBATCH -p standby
#
#SBATCH --ntasks=1

which python

psrlist="J1909-3744 J2317+1439 J2043+1711 J1600-3053 J1918-0642 J0613-0200 J1944+0907 J1744-1134 J1910+1256 J0030+0451"
# psrlist="J1909-3744"

for psr in $psrlist; do python injection.py -n $psr -crn -gwecc -psrterm; done

echo "All pulsar injection done."
