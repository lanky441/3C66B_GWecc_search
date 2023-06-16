import numpy as np
import json
import glob
import argparse

from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import signal_base

from enterprise_gwecc import gwecc_target_block, PsrDistPrior
from juliacall import Main as jl

import corner
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--setting", default="irn_crn_gwecc_psrterm.json")

args = parser.parse_args()
setting_file = args.setting
setting = json.load(open(f"{setting_file}", "r"))

datadir = setting["datadir"]
target_params = json.load(open(setting["target_params"], "r"))
psrdist_info = json.load(open(setting["psrdist_info"], "r"))

name = setting["name"]

priors = {
    "tref": target_params["tref"],
    "cos_gwtheta": target_params["cos_gwtheta"],
    "gwphi": target_params["gwphi"],
    "gwdist": target_params["gwdist"],
    "psi": parameter.Uniform(0.0, np.pi)(f"{name}_psi"),
    "cos_inc": parameter.Uniform(-1, 1)(f"{name}_cos_inc"),
    "eta": parameter.Uniform(0.001, 0.25)(f"{name}_eta"),
    "log10_F": target_params["log10_F"],
    "e0": parameter.Uniform(0.001, 0.9)(f"{name}_e0"),
    "gamma0": parameter.Uniform(0.0, np.pi)(f"{name}_gamma0"),
    "gammap": 0.0,
    "l0": parameter.Uniform(0.0, 2 * np.pi)(f"{name}_l0"),
    "lp": 0.0,
    "log10_A": parameter.Uniform(-12, -6)(f"{name}_log10_A"),
    "psrdist": PsrDistPrior(psrdist_info),
}

parfile = list(sorted(glob.glob(f"{datadir}par/*gls.par")))[0]
timfile = list(sorted(glob.glob(f"{datadir}tim/*.tim")))[0]

psr = Pulsar(parfile, timfile)

efac = parameter.Constant(1)
ef = white_signals.MeasurementNoise(efac=efac)
tm = gp_signals.TimingModel(use_svd=True)
wf = gwecc_target_block(
    **priors, spline=True, psrTerm=False, name="gwecc"
)

s = ef + tm + wf

pta = signal_base.PTA([s(psr)])
print(pta.param_names)

def gwecc_target_prior_my(pta, gwdist, tref, tmax, log10_F, name="gwecc"):
    def gwecc_target_prior_fn(params):
        param_map = pta.map_params(params)
        
        log10_A = param_map[f"{name}_log10_A"]
        eta = param_map[f"{name}_eta"]
        e0 = param_map[f"{name}_e0"]
        
        pta_prior = pta.get_lnprior(param_map)
        
        if pta_prior == -np.inf:
            return pta_prior
        elif jl.validate_params_target(
            log10_A,
            eta,
            log10_F,
            e0,
            gwdist,
            tref,
            tmax,
        ):
            return pta_prior
        else:
            return -np.inf

    return gwecc_target_prior_fn

tmax = psr.toas.max()
get_lnprior = gwecc_target_prior_my(
    pta,
    target_params["gwdist"],
    target_params["tref"],
    tmax,
    log10_F=target_params["log10_F"],
    name=name,
)

prior_samples = np.array([[par.sample() for par in pta.params] for _ in range(100000)])
validated_prior_samples = np.array([p for p in prior_samples if np.isfinite(get_lnprior(p))])

corner.corner(validated_prior_samples[:,[1,2,5]], labels=pta.param_names)
plt.show()