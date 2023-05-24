import numpy as np
import json
import corner
import os
import glob
import argparse

from matplotlib import pyplot as plt
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.signals.signal_base import PTA
from enterprise_gwecc import gwecc_target_block, PsrDistPrior, gwecc_target_prior

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import blocks
from enterprise_extensions.sampler import JumpProposal as JP
from enterprise_extensions.sampler import group_from_params

from juliacall import Main as jl
import juliacall


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--niter", default=1e6)
parser.add_argument("-d", "--datadir", default="../data/")
parser.add_argument("-c", "--chaindir", default="chains/")
parser.add_argument("-hc", "--hotchains", action='store_true')

args = parser.parse_args()

datadir = args.datadir
Niter = args.niter
chaindir = args.chaindir

add_jumps = True
hotchains = True
resume = True


psrlist = ["J1909-3744", "J2317+1439", "J2043+1711", "J1600-3053", "J1918-0642", 
           "J0613-0200", "J1944+0907", "J1744-1134", "J1910+1256", "J0030+0451"]


def get_ew_groups(pta, name='gwecc'):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    params = pta.param_names
    ndim = len(params)
    groups = [list(np.arange(0, ndim))]
    #groups = []

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    print(f"snames = {snames}")
    
    if 'red noise' in snames:

        # create parameter groups for the red noise parameters
        rnpsrs = [p.split('_')[0] for p in params if 'red_noise_log10_A' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])

    if f'{name}_e0' in params:
        gpars = [x for x in params if name in x] #global params
        groups.append([params.index(gp) for gp in gpars]) #add global params

        #pair global params
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_eta')]])
        # groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_e0')]])
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_cos_inc')]])
        # groups.extend([[params.index(f'{name}_eta'), params.index(f'{name}_cos_inc')]])
        # groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_l0')]])
        groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_psi')]])
        # groups.extend([[params.index(f'{name}_psi'), params.index(f'{name}_l0')]])

    if 'gwb_gamma' in params:
        groups.extend([[params.index('gwb_gamma'), params.index('gwb_log10_A')]])
    
    psrdist_params = [ p for p in params if 'psrdist' in p ]
    lp_params = [ p for p in params if 'lp' in p ]
    gammap_params = [ p for p in params if 'gammap' in p ]
    
    for pd, lp, gp in zip (psrdist_params, lp_params, gammap_params):
        groups.extend([[params.index(pd), params.index(lp), params.index(gp), 
                                params.index(f'{name}_log10_A')]])
        
    return groups

true_params = json.load(open(f"{datadir}/true_gwecc_params.dat", "r"))
psrdist_info = json.load(open(f"{datadir}/pulsar_distances_12p5.json", "r"))

empirical_distr = datadir + 'empirical_distributions.pkl'

name = "gwecc"
priors = {
    "tref": true_params["tref"],
    "cos_gwtheta": true_params["cos_gwtheta"],
    "gwphi": true_params["gwphi"],
    "gwdist": true_params["gwdist"],
    "psi": Uniform(0, np.pi)(f"{name}_psi"),
    "cos_inc": Uniform(-1, 1)(f"{name}_cos_inc"),
    "eta": Uniform(0.001, 0.25)(f"{name}_eta"),
    "log10_F": true_params["log10_F"],
    "e0": Uniform(0.001, 0.9)(f"{name}_e0"),
    "gamma0": Uniform(0, np.pi)(f"{name}_gamma0"),
    "gammap": Uniform(0, np.pi),
    "l0": Uniform(0.0,2*np.pi)(f"{name}_l0"),
    "lp": Uniform(0.0,2*np.pi),
    "log10_A": Uniform(-11, -5)(f"{name}_log10_A"),
    "psrdist": PsrDistPrior(psrdist_info),
}

parfiles = sorted(glob.glob(datadir + 'partim/*dmxset.par'))
timfiles = sorted(glob.glob(datadir + 'partim/*.tim'))

if psrlist is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist]

print(parfiles, timfiles)

psrs = []

for par, tim in zip(parfiles, timfiles):
    psr = Pulsar(par, tim)
    psrs.append(psr)

for psr in psrs:
    print(psr.name)


tmax = max([np.max(p.toas) for p in psrs])
tmin = min([np.min(p.toas) for p in psrs])
Tspan = tmax - tmin
print('tmax = MJD ', tmax/86400)
print('Tspan = ', Tspan/const.yr, 'years')

tm = MarginalizingTimingModel(use_svd=True)

# wn = MeasurementNoise(efac=1)
# make WN signal (now with ECORR!)
wn_ec = blocks.white_noise_block(vary=False, inc_ecorr=True, select="backend")

rn = blocks.red_noise_block(psd="powerlaw", components=30,Tspan=Tspan)

crn = blocks.common_red_noise_block(prior='log-uniform', name='gwb', Tspan=Tspan)

wf = gwecc_target_block(**priors, spline=True, psrTerm=True, tie_psrTerm=False, name='')

signal = tm + wn_ec + rn + crn + wf


models = []
for p in psrs:
    models.append(signal(p))


pta = PTA(models)
print(pta.params)
# print(pta.summary())

nfile = f"{datadir}/channelized_12p5yr_v3_full_noisedict.json"
with open(nfile, "r") as f:
    noisedict = json.load(f)

# set the fixed WN params
pta.set_default_params(noisedict)
print(pta.summary())


def gwecc_target_prior_my(pta, gwdist, tref, tmax, log10_F, name="gwecc"):
    def gwecc_target_prior_fn(params):
        param_map = pta.map_params(params)
        if jl.validate_params_target(
            param_map[f"{name}_log10_A"],
            param_map[f"{name}_eta"],
            log10_F,
            param_map[f"{name}_e0"],
            gwdist,
            tref,
            tmax,
        ):
            return pta.get_lnprior(param_map)
        else:
            return -np.inf

    return gwecc_target_prior_fn

get_lnprior = gwecc_target_prior_my(pta, true_params["gwdist"], true_params["tref"], tmax,
                                    log10_F=true_params["log10_F"], name=name)


# def gwecc_target_likelihood_my(pta):
#     def gwecc_target_likelihood_fn(params):
#         param_map = pta.map_params(params)
#         try:
#             lnlike = pta.get_lnlikelihood(param_map)
#         except juliacall.JuliaError:
#             print(juliacall.JuliaError)
#             lnlike = -np.inf
#         return lnlike
#     return gwecc_target_likelihood_fn

# get_lnlikelihood = gwecc_target_likelihood_my(pta)



# comm_params = [x for x in pta.param_names if name in x]
# print(f'gwecc params = {comm_params}')
# n_cmnparams = len(comm_params)

# x_true = []
# ndim = len(pta.param_names)
# for i in range(ndim):
#     if i < ndim - n_cmnparams:
#         x_true.append(noisedict[pta.param_names[i]])
#     else:
#         # x_true.append(true_params[pta.param_names[i][(len(name) + 1) :]])
#         x_true.append(pta.params[i].sample())
# print(x_true)
# print("Log-likelihood at true params is", pta.get_lnlikelihood(x_true))


lnprior_x0 = -np.inf
while lnprior_x0 == -np.inf:
    x0 = [p.sample() for p in pta.params]
    lnprior_x0 = get_lnprior(x0)
    print(f'lnprior(x0) = {get_lnprior(x0)}')
print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))



groups = get_ew_groups(pta, name=name)
print(f'groups = {groups}')


ndim = len(x0)
x0 = np.hstack(x0)

cov = np.diag(np.ones(ndim) * 0.1**2)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, get_lnprior, cov, groups=groups,
                 outDir=chaindir, resume=resume)


if add_jumps:
    jp = JP(pta, empirical_distr=empirical_distr)
    
#     if 'red noise' in jp.snames:
#         sampler.addProposalToCycle(jp.draw_from_red_prior, 20)
    if empirical_distr:
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 20)
    
    sampler.addProposalToCycle(jp.draw_from_prior, 30)

    # draw from ewf priors
    ew_params = [x for x in pta.param_names if name in x]
    for ew in ew_params:
        sampler.addProposalToCycle(jp.draw_from_par_prior(ew),5)

sampler.sample(x0, Niter, writeHotChains=hotchains)

print("Sampler run completed successfully.")

