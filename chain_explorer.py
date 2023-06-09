import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_file", default="all_chains/chains_earth/chain_1.txt")
parser.add_argument("-noise", "--plot_noise_params", action='store_true')
parser.add_argument("-psrterm", "--psrterm", action='store_true')


args = parser.parse_args()
chain_file = args.chain_file
plot_noise_params = args.plot_noise_params
psrterm = args.psrterm


chain_folder = "/".join(chain_file.split("/")[:-1]) + "/"
param_names = np.genfromtxt(f"{chain_folder}/params.txt", dtype=str)
psrlist = np.genfromtxt(f"{chain_folder}/psrlist.txt", dtype=str)
noise_params_med = json.load(open(f"data/noise_param_median_5f.json", "r"))
npsr = len(psrlist)

chain = np.loadtxt(chain_file)
print(chain.shape)

burn = chain.shape[0] //2


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
