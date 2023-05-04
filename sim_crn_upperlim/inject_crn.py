import json
import os
import glob
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import libstempo as lst
import libstempo.plot as lstplot
import libstempo.toasim as toasim

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--psrnumber", type=int)
parser.add_argument("-d", "--datadir", default='../data/')
parser.add_argument("-o", "--outdir", default='simulated_partim/')
args = parser.parse_args()

num = args.psrnumber
datadir = args.datadir
output_dir = args.outdir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(f"Working with pulsar no. {num}")

# choosing par and tim file to work with
parfile = sorted(glob.glob(datadir + '/partim/*dmxset.par'))[num]
timfile = sorted(glob.glob(datadir + '/partim/*.tim'))[num]

print(parfile, timfile)


def run_in_shell(cmd):
    print('"""""\nRUNNING: ', cmd, '\n"""""')
    subprocess.call(cmd,shell=True)

# function to save new par and tim files
def save_psr_sim(psr, savedir):
    print("Writing simulated data for", psr.name)
    psr.savepar(f"{savedir}/{psr.name}_simulate.par")
    psr.savetim(f"{savedir}/{psr.name}_simulate.tim")
    lst.purgetim(f"{savedir}/{psr.name}_simulate.tim")

    return(f"{savedir}/{psr.name}_simulate.par", f"{savedir}/{psr.name}_simulate.tim")


if not os.path.exists(f"{output_dir}/plots"):
    os.mkdir(f"{output_dir}/plots")


# loading libstempo pulsar object
psr = lst.tempopulsar(parfile=parfile, timfile=timfile,  maxobs=60000)
print(f'Libstempo object {psr.name} loaded.')

# residual plot of the original pulsar
lstplot.plotres(psr, label="Residuals")
plt.title(f"{psr.name} original residuals")
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_original_residuals.pdf')
#plt.show()
plt.clf()

# set all residuals to zero
toasim.make_ideal(psr)
# psr.stoas[:] -= psr.residuals() / 86400.0

toasim.add_efac(psr,efac=1.0)

# add common red noise
toasim.add_gwb(psr, flow=1e-9, fhigh=1e-7, gwAmp=2.4e-15)


# save new par and tim file after injections
sim_par, sim_tim = save_psr_sim(psr, output_dir)

# tempo2 commands to run for fitting
cmd1 = f"tempo2 -newpar -f {sim_par} {sim_tim} -nobs 60000"
cmd2 = f"tempo2 -newpar -f new.par {sim_tim} -nobs 60000"

# tempo2 fitting after injection
run_in_shell(cmd1)
#run_in_shell(cmd2)
#run_in_shell(cmd2)
print('Completed tempo2 fitting.')

# moving the fitted par file to correct location
run_in_shell(f'mv ./new.par {sim_par}')
print(f'Fitted par file {sim_par} written')

# loading the pulsar with simulated par and tim files
psrnew = lst.tempopulsar(parfile=sim_par, timfile=sim_tim,  maxobs=60000)

# residual plot with final par and tim files
lstplot.plotres(psrnew, label="Residuals")
plt.plot(psrnew.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
plt.title(f'{psr.name} fitted residuals')
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_fitted_residuals.pdf')
#plt.show()
plt.clf()

print(f'{psr.name} Done!')
