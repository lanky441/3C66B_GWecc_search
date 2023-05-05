"""
This code loads all the pulsars for which the par and time files are available in 
a working directory into enterprise pulsar objects and append them to a list. Then
it creates a PTA signal model with linearized timing model, fixed WN (efac only), 
a common uncorrelated red noise for all pulsars. After that it constructs a 
ptmcmc sampler object to find the values of free parameters
and runs the sampler.
"""

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

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import blocks
from enterprise_extensions.sampler import JumpProposal as JP
from enterprise_extensions.sampler import group_from_params


workdir = "simulated_partim/"
Niter = 1e6
resume = False
chaindir = "chains_crn/"

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

tm = MarginalizingTimingModel(use_svd=True) #use_svd?

wn = MeasurementNoise(efac=1)

crn = blocks.common_red_noise_block(prior='log-uniform', name='gwb', Tspan=Tspan)

signal = tm + wn + crn 


models = []
for p in psrs:
    models.append(signal(p))


pta = PTA(models)
print(pta.param_names)
print(pta.summary())


x_true = [13/3, np.log10(2.4e-15)]
print(x_true)
print("Log-likelihood at true params is", pta.get_lnlikelihood(x_true))


if x0_close_to_true_params:
    x0 = x_true
    print("Log-likelihood at", x0, "is", pta.get_lnlikelihood(x0))
else:
    x0 = [p.sample() for p in pta.params]
    lnlike_x0 = pta.get_lnlikelihood(x0)
    print("Log-likelihood at", x0, "is", lnlike_x0)
    print(f'lnprior(x0) = {pta.get_lnprior(x0)}')


ndim = len(x0)
x0 = np.hstack(x0)

cov = np.diag(np.ones(ndim) * 0.01**2)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov,
                 outDir=chaindir, resume=resume)

sampler.sample(x0, Niter)

print("Sampler run completed successfully.")

