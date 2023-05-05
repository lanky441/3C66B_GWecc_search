"""
This code loads a pulsar object using a par and a tim files, makes the pulsar
ideal (by substracting the residual from the TOAs). Then it injects a WN with
efac=1, a spatiallly uncorrelated GWB with amplitude 2.4e-15, and a gwecc signal
from a 3C66B like source assuming it is 10 times closer to us. Then it fits the
par file with the TOAs after the injection, and save the fitted par file and tim file.
"""

import json
import os
import glob
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

import libstempo as lst
import libstempo.plot as lstplot
import libstempo.toasim as toasim

import enterprise_gwecc as gwecc
from juliacall import Main as jl
from enterprise.pulsar import Pulsar

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

# reference time
day_to_s = 24 * 3600
tref = Time('2003-01-01', format='isot', scale='utc')
print(f'Reference time = MJD {tref.mjd}')

# sky position of 3C66B
RA, DEC = ('02h23m11.4112s', '+42d59m31.385s')
pos = SkyCoord(RA, DEC, frame='icrs')
print(f'Sky position of the GW source = {pos}')

# mass, period, distance, and eccentricity of the source
m1, m2 = (1.2e9, 7.0e8)
P0 = (1.05 * u.year).to(u.s).value   #s
dL = 9.31   #Mpc
ecc0 = 0.3

m = m1 + m2
log10_m = np.log10(m)
eta = m1*m2/m**2
log10F = np.log10(2/P0)

mass = jl.mass_from_log10_mass(log10_m, eta)
n0 = jl.mean_motion_from_log10_freq(log10F)
e0 = jl.Eccentricity(ecc0)
gwdist = jl.dl_from_gwdist(dL)

print(mass, n0, e0, gwdist)

# calculating amplitude from binary parameters
ampl = jl.gw_amplitude(mass, n0, e0, gwdist)
log10_A = np.log10(ampl/n0.n)
print(f'log10_A = {log10_A}')

# checking consistency of mass estimation from amplitude
print(f'Mass estimated from log10A and distance: {jl.mass_from_gwdist(log10_A, log10F, ecc0, dL, eta)}')

# all source parameters
gwecc_params = {
    "cos_gwtheta": np.cos(np.pi/2 - pos.dec.radian),
    "gwphi": pos.ra.radian,
    "psi": np.pi/8,
    "cos_inc": np.cos(np.pi/3),
    "eta": eta,
    "log10_F": log10F,
    "e0": ecc0,
    "gamma0": np.pi/3,
    "gammap": 0.0,
    "l0": np.pi/2,
    "lp": 0.0,
    "tref": tref.mjd * day_to_s,
    "log10_A": log10_A,
    "gwdist": dL,
}

# storing the source parameters
with open(f"{output_dir}/true_gwecc_params.dat", "w") as outfile:
    json.dump(gwecc_params, outfile, indent=4)


# function to add gwecc signal to the pulsar
def add_gwecc(psr, psr_ent, gwecc_params, psrTerm=False):

    signal = (
        np.array(
            gwecc.eccentric_pta_signal_target(
                toas=psr_ent.toas,
                theta = psr_ent.theta,
                phi = psr_ent.phi,
                pdist = psr_ent.pdist,
                psrTerm=psrTerm,
                **gwecc_params,
            )
        )
        / day_to_s
    )

    psr.stoas[:] += signal

    return signal

if not os.path.exists(f"{output_dir}/plots"):
    os.mkdir(f"{output_dir}/plots")


# loading libstempo pulsar object
psr = lst.tempopulsar(parfile=parfile, timfile=timfile,  maxobs=60000)
print(f'Libstempo object {psr.name} loaded.')

# loading enterprise pulsar object
psr_ent = Pulsar(parfile, timfile)
print(f'Enterprise Pulsar object {psr_ent.name} loaded.')
print(f'RMS = {psr.rms()*1e6}, chi-sq = {psr.chisq()}')

# residual plot of the original pulsar
lstplot.plotres(psr, label="Residuals")
plt.title(f"{psr.name} original residuals")
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_original_residuals.pdf')
#plt.show()
plt.clf()

# set all residuals to zero
toasim.make_ideal(psr)

toasim.add_efac(psr,efac=1.0)

# add common red noise
toasim.add_gwb(psr, flow=1e-9, fhigh=1e-7, gwAmp=2.4e-15)

# add gwecc signal
signal = add_gwecc(psr, psr_ent, gwecc_params)
print("Simulated TOAs for", psr.name)

# residual plot after injecting the signals
lstplot.plotres(psr, label="Residuals")
plt.plot(psr.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
plt.title(f'{psr.name} injected signal')
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_injected_signal.pdf')
#plt.show()
plt.clf()

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
