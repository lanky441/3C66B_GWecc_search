import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_folder", default="all_chains/chains_earth/")
parser.add_argument("-nc", "--plot_noise_chains", action='store_true')
parser.add_argument("-np", "--plot_noise_posterior", action='store_true')
parser.add_argument("-compare", "--compare_irn_with_NG12p5", action='store_true')
parser.add_argument("-pc", "--plot_psrdist_chains", action='store_true')
parser.add_argument("-pp", "--plot_psrdist_posterior", action='store_true')
parser.add_argument("-gwb", "--plot_gwb_params", action='store_true')
parser.add_argument("-3p", "--plot_A_e_eta", action='store_true')
parser.add_argument("-ip", "--plot_invalid_param", action='store_true')
parser.add_argument("-b", "--burn_fraction", default=0.25)


args = parser.parse_args()
chain_folder = args.chain_folder
plot_noise_chains = args.plot_noise_chains
plot_noise_posterior = args.plot_noise_posterior
comp_NG = args.compare_irn_with_NG12p5
plot_psrdist_chains = args.plot_psrdist_chains
plot_psrdist_posterior = args.plot_psrdist_posterior
plot_gwb_params = args.plot_gwb_params
plot_A_e_eta = args.plot_A_e_eta
plot_invalid_param = args.plot_invalid_param
burn_frac = float(args.burn_fraction)


if os.path.isfile(f"{chain_folder}/chain_1.txt"):
    chain_file = f"{chain_folder}/chain_1.txt"
elif os.path.isfile(f"{chain_folder}/chain_1.0.txt"):
    chain_file = f"{chain_folder}/chain_1.0.txt"
else:
    sys.exit("Could not find any chain file in the chain folder! Exiting!")

# chain_folder = "/".join(chain_file.split("/")[:-1]) + "/"
param_names = np.genfromtxt(f"{chain_folder}/params.txt", dtype=str)
psrlist = np.genfromtxt(f"{chain_folder}/psrlist.txt", dtype=str)
noise_params_med = json.load(open(f"data/noise_param_median_5f.json", "r"))
npsr = len(psrlist)

chain = np.loadtxt(chain_file)
print(f"Chain shape = {chain.shape}")

burn = int(chain.shape[0] * burn_frac)

if comp_NG:
    NG12p5chain = np.genfromtxt('data/NG12p5_chain_5f_free_gamma_thinned.txt')
    print(f"NG12p5 chain shape = {NG12p5chain.shape}")

    # Reading the names of the parameters present in the chain
    NG12p5params = np.genfromtxt('data/NG12p5_chain_5f_free_gamma_params.txt', dtype='str')

if plot_noise_chains:
    for psrnum, psr in enumerate(psrlist):
        pltnum = psrnum%16
        
        plt.subplot(8,4,2*pltnum+1)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_red_noise_gamma")[0]], ls='', color='C0', 
                 marker='.', alpha=0.1, label=f"{psr}_gamma")
        plt.axhline(noise_params_med[f"{psr}_red_noise_gamma"], c="red")
        plt.legend(loc=3)
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([]) 
        
        plt.subplot(8,4,2*pltnum+2)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_red_noise_log10_A")[0]], ls='', color='C2', 
                 marker='.', alpha=0.1, label=f"{psr}_log10_A")
        plt.axhline(noise_params_med[f"{psr}_red_noise_log10_A"], c="red")
        plt.legend(loc=3)
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([]) 
        
        if (psrnum+1)%16 == 0 or psrnum == len(psrlist)-1:
            plt.show()

if plot_noise_posterior:
    for psrnum, psr in enumerate(psrlist):
        pltnum = psrnum%16
        
        plt.subplot(8,4,2*pltnum+1)
        plt.hist(chain[burn:, np.where(param_names == f"{psr}_red_noise_gamma")[0]], bins=25, density=True, 
                 color='C0', label=f"{psr}_gamma", histtype='step')
        if comp_NG:
            plt.hist(NG12p5chain[:, np.where(NG12p5params == f"{psr}_red_noise_gamma")[0]], bins=25, density=True, 
                 color='C1', label=f"NG", histtype='step')
        plt.legend(loc=2)
        
        plt.subplot(8,4,2*pltnum+2)
        plt.hist(chain[burn:, np.where(param_names == f"{psr}_red_noise_log10_A")[0]], bins=25, density=True,
                 color='C2', label=f"{psr}_log10_A", histtype='step')
        if comp_NG:
            plt.hist(NG12p5chain[:, np.where(NG12p5params == f"{psr}_red_noise_log10_A")[0]], bins=25, density=True, 
                 color='C3', label=f"NG", histtype='step')
        plt.legend(loc=2)
        
        if (psrnum+1)%16 == 0 or psrnum == len(psrlist)-1:
            plt.show()

if plot_psrdist_chains:
    for psrnum, psr in enumerate(psrlist):
        pltnum = psrnum%16
        
        plt.subplot(8,6,3*pltnum+1)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_gammap")[0]], ls='', color='C0', 
                 marker='.', alpha=0.1, label=f"{psr}_gammap")
        plt.legend(loc=3)
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([]) 
        
        plt.subplot(8,6,3*pltnum+2)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_lp")[0]], ls='', color='C2', 
                 marker='.', alpha=0.1, label=f"{psr}_lp")
        plt.legend(loc=3)
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([])
            
        plt.subplot(8,6,3*pltnum+3)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_psrdist")[0]], ls='', color='C3', 
                 marker='.', alpha=0.1, label=f"{psr}_psrdist")
        plt.legend(loc=3)
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([])
        
        if (psrnum+1)%16 == 0 or psrnum == len(psrlist)-1:
            plt.show()

if plot_psrdist_posterior:
    for psrnum, psr in enumerate(psrlist):
        pltnum = psrnum%16
        
        plt.subplot(8,6,3*pltnum+1)
        plt.hist(chain[burn:, np.where(param_names == f"{psr}_gammap")[0]], bins=25, density=True, 
                 color='C0', label=f"{psr}_gammap")
        plt.legend(loc=2)
        
        plt.subplot(8,6,3*pltnum+2)
        plt.hist(chain[burn:, np.where(param_names == f"{psr}_lp")[0]], bins=25, density=True,
                 color='C2', label=f"{psr}_lp")
        plt.legend(loc=2)
        
        plt.subplot(8,6,3*pltnum+3)
        plt.hist(chain[burn:, np.where(param_names == f"{psr}_psrdist")[0]], bins=25, density=True,
                 color='C3', label=f"{psr}_psrdist")
        plt.legend(loc=2)
        
        if (psrnum+1)%16 == 0 or psrnum == len(psrlist)-1:
            plt.show()
            
            
gwb_ecw_params = [p for p in param_names  if 'gwb' in p or 'gwecc' in p]
ndim = len(gwb_ecw_params)
print(f"Common parameters = {gwb_ecw_params}")

for i, param in enumerate(gwb_ecw_params):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[:, np.where(param_names == param)[0]], ls='', marker='.', alpha=0.1)
    plt.ylabel("_".join(param.split("_")[1:]))
plt.show()

for i, param in enumerate(gwb_ecw_params):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[burn:, np.where(param_names == param)[0]], ls='-', marker=None, alpha=1.0)
    plt.ylabel("_".join(param.split("_")[1:]))
plt.show()

if plot_gwb_params:
    gwb_gamma_idx = np.where(param_names == "gwb_gamma")[0][0]

    figure = corner.corner(chain[burn:, gwb_gamma_idx:gwb_gamma_idx+2], labels=["gwb_gamma", "gwb_log10_A"],
                           color='C0', plot_contours=False, hist_kwargs={"density":True})
    if comp_NG:
        gwb_gamma_NG_idx = np.where(NG12p5params == "gwb_gamma")[0][0]
        corner.corner(NG12p5chain[:, gwb_gamma_NG_idx:gwb_gamma_NG_idx+2], fig=figure, 
                      labels=["gwb_gamma", "gwb_log10_A"], color='C1', plot_contours=False, hist_kwargs={"density":True})
    plt.show()

plt.plot(chain[burn:, -3])
plt.ylabel('log_likelihood')
# plt.title(f'last log_likelihood = {chain[-1, -3]}')
plt.show()

if plot_A_e_eta:
    e_idx = np.where(param_names == "gwecc_e0")[0][0]
    eta_idx = np.where(param_names == "gwecc_eta")[0][0]
    A_idx = np.where(param_names == "gwecc_log10_A")[0][0]
    figure2 = corner.corner(chain[burn:, [e_idx, eta_idx, A_idx]], labels=["e0", "eta", "log10_A"],
                           color='C0', plot_contours=False, hist_kwargs={"density":True})
    
    if plot_invalid_param:
        invalid_params = np.genfromtxt(f"{chain_folder}/invalid_params.txt", dtype=None, usecols = (0, 1, 2))
        corner.corner(invalid_params[:, [2, 1, 0]], fig=figure2, 
                     color='C1', plot_contours=False, hist_kwargs={"density":True})
    
    plt.show()
    

corner.corner(chain[burn:,-ndim-4:-4], labels=["_".join(p.split("_")[1:]) for p in gwb_ecw_params],  plot_contours=False)
plt.show()
