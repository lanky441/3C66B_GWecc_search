"""
This code loads a pulsar object using a par and a tim files, makes the pulsar
ideal (by substracting the residual from the TOAs). Then it injects a WN with
efac=1, individual pulsar red noise with parameters from noise dict, and a gwecc signal
from a 3C66B like source assuming it is 10 times closer to us. Then it fits the
par file with the TOAs after the injection, and save the fitted par file and tim file.
"""

import json
import os
import glob
import argparse
import subprocess
import sys

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
parser.add_argument("-n", "--psrname", required=True)
parser.add_argument("-d", "--datadir", default='../data')
parser.add_argument("-o", "--outdir", default='simulated_partim/')

parser.add_argument("-gwecc", "--inject_gwecc", action='store_true')
parser.add_argument("-psrterm", "--include_psrterm", action='store_true')
parser.add_argument("-irn", "--inject_irn", action='store_true')
parser.add_argument("-crn", "--inject_crn", action='store_true')
parser.add_argument("-wn", "--wn_from_dict", action='store_true')

args = parser.parse_args()

psrname = args.psrname
datadir = args.datadir
output_dir = args.outdir

inject_gwecc = args.inject_gwecc
include_psrterm = args.include_psrterm
inject_irn = args.inject_irn
inject_crn = args.inject_crn
wn_from_dict = args.wn_from_dict

if include_psrterm and not inject_gwecc:
    sys.exit("Cannot include psrterm when not injecting gwecc signal.")

gwb_log10_A = 2.4e-15


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(f"Working with PSR {psrname}")

parfile = f'{datadir}/partim/{psrname}_dmxset.par'
timfile = f'{datadir}/partim/{psrname}_NANOGrav_12yv4.tim'


# loading libstempo pulsar object
psr = lst.tempopulsar(parfile=parfile, timfile=timfile, maxobs=1e5)
print(f'Libstempo object {psr.name} loaded.')

# loading enterprise pulsar object
psr_ent = Pulsar(parfile, timfile)
print(f'Enterprise Pulsar object {psr_ent.name} loaded.')


if not os.path.exists(f"{output_dir}/plots"):
    os.mkdir(f"{output_dir}/plots")

# residual plot of the original pulsar
lstplot.plotres(psr, label="Residuals")
plt.title(f"{psr.name}, RMS = {psr.rms()*1e6} us")
plt.legend()
plt.savefig(f'{output_dir}/plots/{psr.name}_original_residuals.pdf')
#plt.show()
plt.clf()


# function to save new par and tim files
def save_psr_sim(psr, savedir):
    print("Writing simulated data for", psr.name)
    psr.savepar(f"{savedir}/{psr.name}_simulate.par")
    psr.savetim(f"{savedir}/{psr.name}_simulate.tim")
    lst.purgetim(f"{savedir}/{psr.name}_simulate.tim")

    return(f"{savedir}/{psr.name}_simulate.par", f"{savedir}/{psr.name}_simulate.tim")


##
# SMBHB paremeters for gwecc injection
##

# reference time
day_to_s = 24 * 3600
tref = Time('2003-01-01', format='isot', scale='utc')

# sky position of 3C66B
RA, DEC = ('02h23m11.4112s', '+42d59m31.385s')
pos = SkyCoord(RA, DEC, frame='icrs')

# mass, period, distance, and eccentricity of the source
m1, m2 = (1.2e9, 7.0e8)
P0 = (1.05 * u.year).to(u.s).value   #s
dL = 93.1/10   #Mpc
ecc0 = 0.3

m = m1 + m2
log10_m = np.log10(m)
eta = m1*m2/m**2
log10F = np.log10(2/P0)

mass = jl.mass_from_log10_mass(log10_m, eta)
n0 = jl.mean_motion_from_log10_freq(log10F)
e0 = jl.Eccentricity(ecc0)
gwdist = jl.dl_from_gwdist(dL)

# calculating amplitude from binary parameters
ampl = jl.gw_amplitude(mass, n0, e0, gwdist)
log10_A = np.log10(ampl/n0.n)

# checking consistency of mass estimation from amplitude
# print(f'Mass estimated from log10A and distance: {jl.mass_from_gwdist(log10_A, log10F, ecc0, dL, eta)}')

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


# Load the noise parameters for 12.5-yr DR noisedict
if inject_irn or wn_from_dict:
    try:
        with open(f'{datadir}/channelized_12p5yr_v3_full_noisedict.json') as nf:
            noise_params = json.load(nf)
        noise_dict = {}
    except:
        sys.exit("Can't find noise dictionary! Exiting!")
        
if wn_from_dict or inject_irn:
    if wn_from_dict:
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
                if inject_irn:
                    if 'gamma' in ky:
                        noise_dict['RN_gamma'] = noise_params[ky]
                    if 'log10_A' in ky:
                        noise_dict['RN_Amp'] = 10**noise_params[ky]
                        
        noise_dict['equads'] = np.array(noise_dict['equads'])
        noise_dict['efacs'] = np.array(noise_dict['efacs'])
        noise_dict['ecorrs'] = np.array(noise_dict['ecorrs'])    
    
        if len(noise_dict['ecorrs'])==0: #Easier to just delete these dictionary items if no ECORR values. 
            noise_dict.__delitem__('ecorrs')
            
    else:
        for ky in list(noise_params.keys()):
            if psr.name in ky:
                if 'gamma' in ky:
                    noise_dict['RN_gamma'] = noise_params[ky]
                if 'log10_A' in ky:
                    noise_dict['RN_Amp'] = 10**noise_params[ky]
                        
    print(noise_dict)
                


# set all residuals to zero
print("Making the pulsar ideal.")
toasim.make_ideal(psr)

# add white noise
if not wn_from_dict:
    print("Adding efac=1.0 white noise.")
    toasim.add_efac(psr,efac=1.0)
else:
    print("Adding white noise from noise dictionary.")
    # add efacs
    if len(noise_dict['efacs']) > 0:
        toasim.add_efac(psr, efac = noise_dict['efacs'][:,1], 
                    flagid = 'f', flags = noise_dict['efacs'][:,0])

    # add equads
    toasim.add_equad(psr, equad = noise_dict['equads'][:,1], 
                 flagid = 'f', flags = noise_dict['equads'][:,0])

    # add jitter
    # Only NANOGrav Pulsars have ECORR
    try: 
        toasim.add_jitter(psr, ecorr = noise_dict['ecorrs'][:,1], 
                      flagid='f', flags = noise_dict['ecorrs'][:,0], 
                      coarsegrain = 1.0/86400.0)
    except KeyError:
        print(f'Could not add ECORR for {psr.name}!')
        pass

# add individual pulsar red noise
if inject_irn:
    print("Adding individual pulsar red noise.")
    toasim.add_rednoise(psr, noise_dict['RN_Amp'], noise_dict['RN_gamma'], 
                components = 30)
    
# add common red noise
if inject_crn:
    print("Adding common red noise.")
    toasim.add_gwb(psr, flow=1e-9, fhigh=1e-7, gwAmp=gwb_log10_A)
    
    gwb_params = {"log10_A": gwb_log10_A,
                  "gamma": 13/3}
    
    # storing the source parameters
    with open(f"{output_dir}/true_gwb_params.dat", "w") as outfile:
        json.dump(gwb_params, outfile, indent=4)

# add gwecc signal
if inject_gwecc:
    print('Adding gwecc signal.')
    signal = add_gwecc(psr, psr_ent, gwecc_params)

    # residual plot after injecting the signals
    lstplot.plotres(psr, label="Residuals")
    plt.plot(psr.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
    plt.title(f'{psr.name} injected signal')
    plt.legend()
    plt.savefig(f'{output_dir}/plots/{psr.name}_injected_signal.pdf')
    #plt.show()
    plt.clf()

# fit after injection
psr.fit(iters=3)

# save new par and tim file after injections
sim_par, sim_tim = save_psr_sim(psr, output_dir)

# loading the pulsar with simulated par and tim files
psrnew = lst.tempopulsar(parfile=sim_par, timfile=sim_tim,  maxobs=1e5)

# residual plot with final par and tim files
lstplot.plotres(psrnew, label="Residuals")
if inject_gwecc:
    plt.plot(psrnew.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal", zorder=100)
plt.title(f'{psrnew.name}, RMS = {psr.rms()*1e6} us')
plt.legend()
plt.savefig(f'{output_dir}/plots/{psrnew.name}_fitted_residuals.pdf')
#plt.show()
plt.clf()

print(f'{psr.name} Done!')
