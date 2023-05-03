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

# import libstempo as lst
# import libstempo.plot as lstplot
# import libstempo.toasim as toasim

# import enterprise_gwecc as gwecc
# from juliacall import Main as jl
# from enterprise.pulsar import Pulsar

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--psrnumber", type=int)
parser.add_argument("-d", "--datadir", default='../data/partim/')
parser.add_argument("-o", "--outdir", default='simulated_partim/')
args = parser.parse_args()

num = args.psrnumber
datadir = args.datadir
output_dir = args.outdir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(num)

parfile = sorted(glob.glob(datadir + '*dmxset.par'))[num]
timfile = sorted(glob.glob(datadir + '*.tim'))[num]

print(parfile, timfile)

def run_in_shell(cmd):
    print('"""""\nRUNNING: ', cmd, '\n"""""')
    subprocess.call(cmd,shell=True)

def save_psr_sim(psr, savedir):
    print("Writing simulated data for", psr.name)
    psr.savepar(f"{savedir}/{psr.name}_simulate.par")
    psr.savetim(f"{savedir}/{psr.name}_simulate.tim")
    lst.purgetim(f"{savedir}/{psr.name}_simulate.tim")

    return(f"{savedir}/{psr.name}_simulate.par", f"{savedir}/{psr.name}_simulate.tim")

day_to_s = 24 * 3600
tref = Time('2003-01-01', format='isot', scale='utc')
print(f'Reference time = MJD {tref.mjd}')

RA, DEC = ('02h23m11.4112s', '+42d59m31.385s')
pos = SkyCoord(RA, DEC, frame='icrs')
print(f'Sky position of the GW source = {pos}')

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

ampl = jl.gw_amplitude(mass, n0, e0, gwdist)
log10_A = np.log10(ampl/n0.n)
print(f'log10_A = {log10_A}')

print(f'Mass estimated from log10A and distance: {jl.mass_from_gwdist(log10_A, log10F, ecc0, dL, eta)}')

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

with open(f"{output_dir}/true_gwecc_params.dat", "w") as outfile:
    json.dump(gwecc_params, outfile, indent=4)

def add_gwecc(psr, psr_ent, gwecc_params, psrTerm=False):

    # toas = (psr.toas() * day_to_s).astype(float)

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

psr = lst.tempopulsar(parfile=parfile, timfile=timfile,  maxobs=60000)
print(f'Libstempo object {psr.name} loaded.')

psr_ent = Pulsar(parfile, timfile)
print(f'Enterprise Pulsar object {psr_ent.name} loaded.')
print(f'RMS = {psr.rms()*1e6}, chi-sq = {psr.chisq()}')

lstplot.plotres(psr, label="Residuals")
plt.title(f"{psr.name} Residuals")
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_original_residual.pdf')
#plt.show()
plt.clf()

# toasim.make_ideal(psr)
# print(f'Made the pulsar {psr.name} ideal.')
# toasim.add_efac(psr, 1)
signal = add_gwecc(psr, psr_ent, gwecc_params)
print("Simulated TOAs for", psr.name)
