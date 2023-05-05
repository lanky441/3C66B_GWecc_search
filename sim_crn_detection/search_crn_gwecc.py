import numpy as np
import json
import corner
import os
import glob

from matplotlib import pyplot as plt
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.signals.signal_base import PTA
from enterprise_gwecc import gwecc_target_block

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import blocks
from enterprise_extensions.sampler import JumpProposal as JP
from enterprise_extensions.sampler import group_from_params

from juliacall import Main as jl
import juliacall


workdir = "simulated_partim/"
Niter = 1e6
resume = False
chaindir = "chains_single/"

true_params = json.load(open(f"{workdir}/true_gwecc_params.dat", "r"))

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
    "e0": Uniform(0.001, 0.8)(f"{name}_e0"),
    "gamma0": Uniform(0, np.pi)(f"{name}_gamma0"),
    "gammap": 0,
    "l0": Uniform(0.0,2*np.pi)(f"{name}_l0"),
    "lp": 0.0,
    "log10_A": Uniform(-9, -5)(f"{name}_log10_A"),
}

x0_close_to_true_params = False

parfiles = sorted(glob.glob(workdir + '*.par'))
timfiles = sorted(glob.glob(workdir + '*.tim'))

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

wn = MeasurementNoise(efac=1)

crn = blocks.common_red_noise_block(prior='log-uniform', name='gwb', Tspan=Tspan)

wf = gwecc_target_block(**priors, spline=True)

signal = tm + wn + crn + wf


models = []
for p in psrs:
    models.append(signal(p))


pta = PTA(models)
print(pta.param_names)
print(pta.summary())


def gwecc_target_prior_my(pta, gwdist, log10_F, tref, tmax, name="gwecc"):
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

get_lnprior = gwecc_target_prior_my(pta, true_params["gwdist"], true_params["log10_F"],
                                 true_params["tref"], tmax, name=name)


def gwecc_target_likelihood_my(pta):
    def gwecc_target_likelihood_fn(params):
        param_map = pta.map_params(params)
        try:
            lnlike = pta.get_lnlikelihood(param_map)
        except juliacall.JuliaError:
            print(juliacall.JuliaError)
            lnlike = -np.inf
        return lnlike
    return gwecc_target_likelihood_fn

get_lnlikelihood = gwecc_target_likelihood_my(pta)

comm_params = [x for x in pta.param_names if name in x]
print(f'gwecc params = {comm_params}')
n_cmnparams = len(comm_params)

x_true = [13/3, np.log10(2.4e-15)]

ndim = len(pta.param_names)
for i in range(ndim):
    if i < ndim - n_cmnparams:
        print('Already added.')
    else:
        x_true.append(true_params[pta.param_names[i][(len(name) + 1) :]])
        # x_true.append(pta.params[i].sample())
print(x_true)
print("Log-likelihood at true params is", pta.get_lnlikelihood(x_true))


if x0_close_to_true_params:
    x0 = x_true
    print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))
else:
    lnlike_x0 = -np.inf
    while lnlike_x0 == -np.inf:
        x0 = [p.sample() for p in pta.params]
        lnlike_x0 = get_lnlikelihood(x0)
        print("Log-likelihood at", x0, "is", get_lnlikelihood(x0))
        print(f'lnprior(x0) = {get_lnprior(x0)}')



ndim = len(x0)
x0 = np.hstack(x0)

cov = np.diag(np.ones(ndim) * 0.01**2)

sampler = ptmcmc(ndim, get_lnlikelihood, get_lnprior, cov,
                 outDir=chaindir, resume=resume)


#jp = JP(pta)
#sampler.addProposalToCycle(jp.draw_from_prior, 20)


# draw from ewf priors
# ew_params = [x for x in pta.param_names if name in x]
# for ew in ew_params:
#    sampler.addProposalToCycle(jp.draw_from_par_prior(ew),5)

sampler.sample(x0, Niter) #, writeHotChains=True)

print("Sampler run completed successfully.")

