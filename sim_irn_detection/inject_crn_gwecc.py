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

print(num)

parfile = sorted(glob.glob(datadir + '/partim/*dmxset.par'))[num]
timfile = sorted(glob.glob(datadir + '/partim/*.tim'))[num]

print(parfile, timfile)

# Load the noise parameters for 12.5-yr DR noisedict
with open(f'{datadir}/channelized_12p5yr_v3_full_noisedict.json') as nf:
    noise_params = json.load(nf)

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
plt.title(f"{psr.name} original residuals")
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_original_residuals.pdf')
#plt.show()
plt.clf()

noise_dict = {}
noise_dict['equads'] = []
noise_dict['efacs'] = []
noise_dict['ecorrs'] = []

for ky in list(noise_params.keys()):
    if psr.name in ky:
        if 'equad' in ky:
            noise_dict['equads'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
        if 'efac' in ky:
            noise_dict['efacs'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
        if 'ecorr' in ky:
            noise_dict['ecorrs'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
        if 'gamma' in ky:
            noise_dict['RN_gamma'] = noise_params[ky]
        if 'log10_A' in ky:
            noise_dict['RN_Amp'] = 10**noise_params[ky]

noise_dict['equads'] = np.array(noise_dict['equads'])
noise_dict['efacs'] = np.array(noise_dict['efacs'])
noise_dict['ecorrs'] = np.array(noise_dict['ecorrs'])    

if len(noise_dict['ecorrs'])==0: #Easier to just delete these dictionary items if no ECORR values. 
    noise_dict.__delitem__('ecorrs') 

print(f'Noise dict for {psr.name} = {noise_dict}')

#set seed values for the rng
seed_efac = 1234
seed_equad = 5678
seed_jitter = 9101
seed_red = 1121

# set all residuals to zero
psr.stoas[:] -= psr.residuals() / 86400.0

# add efacs
if len(noise_dict['efacs']) > 0:
    toasim.add_efac(psr, efac = noise_dict['efacs'][:,1], 
                flagid = 'f', flags = noise_dict['efacs'][:,0], 
                seed = seed_efac + np.random.randint(47))

# add equads
toasim.add_equad(psr, equad = noise_dict['equads'][:,1], 
             flagid = 'f', flags = noise_dict['equads'][:,0], 
             seed = seed_equad + np.random.randint(47))

# add jitter
# Only NANOGrav Pulsars have ECORR
try: 
    toasim.add_jitter(psr, ecorr = noise_dict['ecorrs'][:,1], 
                  flagid='f', flags = noise_dict['ecorrs'][:,0], 
                  coarsegrain = 1.0/86400.0, seed=seed_jitter + np.random.randint(47))
except KeyError:
    print(f'Could not add ECORR for {psr.name}!')
    pass

## add red noise
toasim.add_rednoise(psr, noise_dict['RN_Amp'], noise_dict['RN_gamma'], 
                components = 30, seed = seed_red + np.random.randint(47))

signal = add_gwecc(psr, psr_ent, gwecc_params)
print("Simulated TOAs for", psr.name)


lstplot.plotres(psr, label="Residuals")
plt.plot(psr.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
plt.title(f'{psr.name} injected signal')
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_injected_signal.pdf')
#plt.show()
plt.clf()

sim_par, sim_tim = save_psr_sim(psr, output_dir)

cmd1 = f"tempo2 -newpar -f {sim_par} {sim_tim} -nobs 60000"
cmd2 = f"tempo2 -newpar -f new.par {sim_tim} -nobs 60000"

run_in_shell(cmd1)
#run_in_shell(cmd2)
#run_in_shell(cmd2)
print('Completed tempo2 fitting.')

run_in_shell(f'mv ./new.par {sim_par}')
print(f'Fitted par file {sim_par} written')

psrnew = lst.tempopulsar(parfile=sim_par, timfile=sim_tim,  maxobs=60000)

lstplot.plotres(psrnew, label="Residuals")
plt.plot(psrnew.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
plt.title(f'{psr.name} fitted residuals')
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_fitted_residuals.pdf')
#plt.show()
plt.clf()

print(f'{psr.name} Done!')
