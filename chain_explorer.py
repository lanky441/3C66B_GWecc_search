import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_folder", default="all_chains/chains_earth/")
parser.add_argument("-noise", "--plot_noise_params", action='store_true')
parser.add_argument("-psrterm", "--psrterm", action='store_true')
parser.add_argument("-b", "--burn_fraction", default=4)


args = parser.parse_args()
chain_folder = args.chain_folder
plot_noise_params = args.plot_noise_params
psrterm = args.psrterm
burn_frac = args.burn_fraction

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
print(chain.shape)

burn = chain.shape[0] //burn_frac


if plot_noise_params:
    for psrnum, psr in enumerate(psrlist):
        pltnum = psrnum%16
        
        plt.subplot(8,4,2*pltnum+1)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_red_noise_gamma")[0]], ls='', marker='.', alpha=0.1)
        plt.axhline(noise_params_med[f"{psr}_red_noise_gamma"], c="red")
        plt.title(f"{psr}_gamma")
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([]) 
        
        plt.subplot(8,4,2*pltnum+2)
        plt.plot(chain[burn:, np.where(param_names == f"{psr}_red_noise_log10_A")[0]], ls='', marker='.', alpha=0.1)
        plt.axhline(noise_params_med[f"{psr}_red_noise_log10_A"], c="red")
        plt.title(f"{psr}_log10_A")
        if pltnum < 14 and psrnum < len(psrlist)-2: plt.xticks([]) 
        
        if (psrnum+1)%16 == 0 or psrnum == len(psrlist)-1:
            plt.show()
    

gwb_ecw_params = [p for p in param_names  if 'gwb' in p or 'gwecc' in p]
ndim = len(gwb_ecw_params)
print(gwb_ecw_params)

for i, param in enumerate(gwb_ecw_params):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[:, np.where(param_names == param)[0]], ls='', marker='.', alpha=0.1)
    plt.ylabel("_".join(param.split("_")[1:]))
plt.show()

for i, param in enumerate(gwb_ecw_params):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[burn:, np.where(param_names == param)[0]], ls='', marker='.', alpha=0.1)
    plt.ylabel("_".join(param.split("_")[1:]))
plt.show()


plt.plot(chain[burn:, -3])
plt.ylabel('log_likelihood')
plt.title(f'last log_likelihood = {chain[-1, -3]}')
plt.show()

# # for i in range(npsr):
# #     corner.corner(chain[burn:, psr_params*i:psr_params*(i+1)])
# #     plt.show()

corner.corner(chain[burn:,-ndim-4:-4], labels=["_".join(p.split("_")[1:]) for p in gwb_ecw_params])
# # plt.savefig("gwecc_sims/plots/corner.pdf")
plt.show()
