import numpy as np
import json
import glob
import argparse
import os


import enterprise
from enterprise import constants as const
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions.sampler import JumpProposal as JP
from enterprise_extensions.sampler import group_from_params
from get_groups import get_ew_groups

from enterprise_gwecc import gwecc_target_block, PsrDistPrior, gwecc_target_prior
from juliacall import Main as jl
import juliacall


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--setting", default="irn_crn_gwecc_psrterm.json")

args = parser.parse_args()
setting = json.load(open(f"{args.setting}", "r"))

datadir = setting["datadir"]
target_params = json.load(open(setting["target_params"], "r"))
psrdist_info = json.load(open(setting["psrdist_info"], "r"))
empirical_distr = setting["empirical_distr"]
nfile = setting["noise_dict"]

psrlist_exclude = setting["psr_exclude"]
psrlist_include = setting["psr_include"]

gamma_vary = setting["gamma_vary"]
name = setting["name"]
psrterm = setting["psrterm"]
tie = setting["tie_psrterm"]

x0_median = setting["x0_median"]
Niter = setting["Niter"]
if os.path.isdir("all_chains"):
    chaindir = "all_chains/" + setting["chaindir"]
else:
    chaindir = setting["chaindir"]
hotchains = setting["write_hotchain"]
resume = setting["resume"]
make_groups = setting["make_groups"]
add_jumps = setting["add_jumps"]



if tie or not psrterm:
    priors = {
        "tref": target_params["tref"],
        "cos_gwtheta": target_params["cos_gwtheta"],
        "gwphi": target_params["gwphi"],
        "gwdist": target_params["gwdist"],
        "psi": parameter.Uniform(0, np.pi)(f"{name}_psi"),
        "cos_inc": parameter.Uniform(-1, 1)(f"{name}_cos_inc"),
        "eta": parameter.Uniform(0.001, 0.25)(f"{name}_eta"),
        "log10_F": target_params["log10_F"],
        "e0": parameter.Uniform(0.001, 0.9)(f"{name}_e0"),
        "gamma0": parameter.Uniform(0, np.pi)(f"{name}_gamma0"),
        "gammap": 0,
        "l0": parameter.Uniform(0.0,2*np.pi)(f"{name}_l0"),
        "lp": 0,
        "log10_A": parameter.Uniform(-11, -5)(f"{name}_log10_A"),
        "psrdist": PsrDistPrior(psrdist_info),
    }
    
if psrterm and not tie:
    priors = {
        "tref": target_params["tref"],
        "cos_gwtheta": target_params["cos_gwtheta"],
        "gwphi": target_params["gwphi"],
        "gwdist": target_params["gwdist"],
        "psi": parameter.Uniform(0, np.pi)(f"{name}_psi"),
        "cos_inc": parameter.Uniform(-1, 1)(f"{name}_cos_inc"),
        "eta": parameter.Uniform(0.001, 0.25)(f"{name}_eta"),
        "log10_F": target_params["log10_F"],
        "e0": parameter.Uniform(0.001, 0.9)(f"{name}_e0"),
        "gamma0": parameter.Uniform(0, np.pi)(f"{name}_gamma0"),
        "gammap": parameter.Uniform(0.0, np.pi),
        "l0": parameter.Uniform(0.0, 2*np.pi)(f"{name}_l0"),
        "lp": parameter.Uniform(0.0, 2*np.pi),
        "log10_A": parameter.Uniform(-11, -5)(f"{name}_log10_A"),
        "psrdist": PsrDistPrior(psrdist_info),
    }


parfiles = sorted(glob.glob(datadir + 'par/*gls.par'))
timfiles = sorted(glob.glob(datadir + 'tim/*.tim'))

if psrlist_exclude is not None:
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0].split('_')[0] not in psrlist_exclude]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0].split('_')[0] not in psrlist_exclude]
    
if psrlist_include !='all':
    parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist_include]
    timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0].split('_')[0] in psrlist_include]

# print(parfiles, timfiles)

psrs = []
ephemeris = 'DE438'

for par, tim in zip(parfiles, timfiles):
    psr = Pulsar(par, tim, ephem=ephemeris)
    psrs.append(psr)

for psr in psrs:
    print(psr.name)
    

with open(nfile, "r") as f:
    noisedict = json.load(f)


# find the maximum time span to set GW frequency sampling
tmin = np.min([p.toas.min() for p in psrs])
tmax = np.max([p.toas.max() for p in psrs])
Tspan = tmax - tmin
print('tmax = MJD ', np.max(tmax)/86400)
print('Tspan = ', Tspan/const.yr, 'years')


# define selection by observing backend
selection = selections.Selection(selections.by_backend)


# white noise parameters
efac = parameter.Constant() 
equad = parameter.Constant() 
ecorr = parameter.Constant() # we'll set these later with the params dictionary

# red noise parameters
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.Uniform(-20, -11)('gwb_log10_A')
if gamma_vary:
    gamma_gw = parameter.Uniform(0, 8)('gwb_gamma')
else:
    gamma_gw = parameter.Constant(4.33)('gwb_gamma')


# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

# gwb (no spatial correlations)
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=cpl, components=5, Tspan=Tspan, name='gwb')

# for spatial correlations you can do...
# spatial correlations are covered in the hypermodel context later
# orf = utils.hd_orf()
# crn = gp_signals.FourierBasisCommonGP(cpl, orf,
#                                       components=30, Tspan=Tspan, name='gw')

# to add solar system ephemeris modeling...
bayesephem=False
if bayesephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

# timing model
tm = gp_signals.TimingModel(use_svd=True)

# eccentric signal
wf = gwecc_target_block(**priors, spline=True, psrTerm=psrterm, tie_psrTerm=tie, name='')

# full model
if bayesephem:
    s = ef + eq + ec + rn + tm + eph + gw + wf
else:
    s = ef + eq + ec + rn + tm + gw + wf


# intialize PTA
models = []
        
for p in psrs:    
    models.append(s(p))
    
pta = signal_base.PTA(models)


# set white noise parameters with dictionary
pta.set_default_params(noisedict)
    

print(pta.params)
# print(pta.summary())

# custom function to get lnprior
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

get_lnprior = gwecc_target_prior_my(pta, target_params["gwdist"], target_params["tref"], tmax,
                                    log10_F=target_params["log10_F"], name=name)

# custom function to get lnlikelihood
def gwecc_target_likelihood_my(pta):
    def gwecc_target_likelihood_fn(params):
        param_map = pta.map_params(params)
        try:
            lnlike = pta.get_lnlikelihood(param_map)
        except juliacall.JuliaError as err_julia:
            print("Domain Error")
            lnlike = -np.inf
        return lnlike
    return gwecc_target_likelihood_fn

get_lnlikelihood = gwecc_target_likelihood_my(pta)


median_params = json.load(open(f"{datadir}/noise_param_median_5f.json", "r"))

# set initial parameters from dict or drawn from prior
x0 = []
if x0_median:
    for p in pta.param_names:
        if "gwecc" in p:
            x0.append(target_params[p])
        else:
            x0.append(median_params[p])
    x0 = np.hstack(x0)
else:
    lnprior_x0 = -np.inf
    while lnprior_x0 == -np.inf:
        x0 = np.hstack([p.sample() for p in pta.params])
        lnprior_x0 = get_lnprior(x0)

print(f"x0 = {x0}")
print(f"lnprior(x0) = {get_lnprior(x0)}")
print(f"lnlikelihood(x0) = {get_lnlikelihood(x0)}")

if make_groups:
    groups = get_ew_groups(pta, name=name)
else:
    groups = None
print(f'groups = {groups}')

ndim = len(x0)

# set up the sampler:
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

sampler = ptmcmc(ndim, get_lnlikelihood, get_lnprior, cov, groups=groups,
                 outDir=chaindir, resume=resume)

# write parameter names
np.savetxt(chaindir + '/params.txt', list(map(str, pta.param_names)), fmt='%s')

if add_jumps:
    jp = JP(pta, empirical_distr=empirical_distr)
    
#     if 'red noise' in jp.snames:
#         sampler.addProposalToCycle(jp.draw_from_red_prior, 20)
    if empirical_distr:
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 30)
    
#     sampler.addProposalToCycle(jp.draw_from_prior, 30)

    # draw from ewf priors
    ew_params = [x for x in pta.param_names if name in x]
    for ew in ew_params:
        sampler.addProposalToCycle(jp.draw_from_par_prior(ew),5)
    
    # draw from gwb priors
    gwb_params = [x for x in pta.param_names if 'gwb' in x]
    for para in gwb_params:
        sampler.addProposalToCycle(jp.draw_from_par_prior(para),5)

sampler.sample(x0, Niter, SCAMweight=25, AMweight=40, DEweight=20, writeHotChains=hotchains)


print("Sampler run completed successfully.")