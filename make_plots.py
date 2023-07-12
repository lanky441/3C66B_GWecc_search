# coding: utf-8

import numpy as np
import wquantiles
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner
import json
import argparse
import os
import sys


def log2linear_weight(value, pmin, pmax):
    mask = np.logical_and(value>=pmin, value<=pmax)
    uniform_prior = 1 / (pmax-pmin)
    linexp_prior = np.log(10) * 10**value / (10**pmax - 10**pmin)
    
    weight = mask * linexp_prior / uniform_prior
    weight /= sum(weight)

    return weight

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_folder", default="all_chains/chains_earth_4_Jun23/")
parser.add_argument("-b", "--burn_fraction", default=0.1)

args = parser.parse_args()
chain_folder = args.chain_folder
burn_frac = float(args.burn_fraction)

if os.path.isfile(f"{chain_folder}/chain_1.txt"):
    chain_file = f"{chain_folder}/chain_1.txt"
elif os.path.isfile(f"{chain_folder}/chain_1.0.txt"):
    chain_file = f"{chain_folder}/chain_1.0.txt"
else:
    print("Could not find any chain file in the chain folder! Exiting!")

param_names = np.genfromtxt(f"{chain_folder}/params.txt", dtype=str)
psrlist = np.genfromtxt(f"{chain_folder}/psrlist.txt", dtype=str)
chain = np.loadtxt(chain_file)
print(f"Chain shape = {chain.shape}")

burn = int(chain.shape[0] * burn_frac)

e_idx = np.where(param_names == "gwecc_e0")[0][0]
eta_idx = np.where(param_names == "gwecc_eta")[0][0]
A_idx = np.where(param_names == "gwecc_log10_A")[0][0]

corner.corner(chain[burn:, [e_idx, eta_idx, A_idx]], labels=["e0", "eta", "log10_A"],
                           color='C0', plot_contours=False, hist_kwargs={"density":True})
plt.show()

es = chain[burn:, e_idx]
etas = chain[burn:, eta_idx]
gwecc_log10_As = chain[burn:, A_idx]

mask = np.logical_and(es<0.5, etas>0.1)
corner.corner(np.array([es[mask], etas[mask], gwecc_log10_As[mask]]).T, labels=["e0", "eta", "log10_A"],
                           color='C0', plot_contours=False, hist_kwargs={"density":True})
plt.show()

# Calculate the weights to convert from log-uniform to uniform posterior
A_post_ws = log2linear_weight(gwecc_log10_As, -12, -6)

# plt.hist(10**gwecc_log10_As, bins=15, weights=A_post_ws)
# plt.show()



ep, etap, logAp, valid = np.genfromtxt("valid_param_space.txt").transpose()

ev = ep[valid==1]
etav = etap[valid==1]
logAv = logAp[valid==1]

# Calculate the weights to convert from log-uniform to uniform prior
A_prior_ws = log2linear_weight(logAv, -12, -6)

# plt.hist(logAv, bins=15, weights=None)
# plt.show()

# print(len(ev), len(etav), len(logAv))

# plt.hist2d(ev, etav, bins=8)
# plt.show()


# Define the number of bins for the first two parameters
num_bins = 8

# Calculate the bin indices for the first two parameters
e_bins = np.linspace(0, 0.8, num_bins + 1)
eta_bins = np.linspace(0, 0.25, num_bins + 1)

# print(e_bins, eta_bins)

# Digitize the first two parameters to obtain the bin indices
e_bin_indices = np.digitize(es, e_bins)
eta_bin_indices = np.digitize(etas, eta_bins)

ev_bin_indices = np.digitize(ev, e_bins)
etav_bin_indices = np.digitize(etav, eta_bins)


# Initialize an empty array to store the percentile values for each bin
percentiles = np.zeros((num_bins, num_bins))
valid_percentiles = np.zeros((num_bins, num_bins))

quantiles = np.zeros((num_bins, num_bins))
valid_quantiles = np.zeros((num_bins, num_bins))


# Calculate the 95th percentile value for each bin
fig = plt.figure(figsize=(12, 12))
for i in range(num_bins):
    for j in range(num_bins):
        # Select the data points that fall within the current bin
        mask = (e_bin_indices == i + 1) & (eta_bin_indices == j + 1)
        # Calculate the 95th percentile of the third parameter for the current bin
        percentiles[i, j] = np.percentile(gwecc_log10_As[mask], 95)
        quantiles[i, j] = wquantiles.quantile(gwecc_log10_As[mask], A_post_ws[mask], 0.95)
        
        mask_valid = (ev_bin_indices == i + 1) & (etav_bin_indices == j + 1)
        valid_percentiles[i, j] = np.percentile(logAv[mask_valid], 95)
        valid_quantiles[i, j] = wquantiles.quantile(logAv[mask_valid], A_prior_ws[mask_valid], 0.95)
        
        plt.subplot(8, 8, 8*i+j+1)
#         plt.hist(10**logAv[mask_valid], bins=10, alpha=0.5, density=True, weights=A_prior_ws[mask_valid])
#         plt.hist(10**gwecc_log10_As[mask], bins=10, alpha=0.5, density=True, weights= A_post_ws[mask])
        plt.hist(logAv[mask_valid], bins=10, alpha=0.5, density=True, weights=None,
                 label=f"{(e_bins[i] + e_bins[i+1])/2:.2f}, {(eta_bins[j] + eta_bins[j+1])/2:.2f} prior")
        plt.hist(gwecc_log10_As[mask], bins=10, alpha=0.5, density=True, weights=None,
                 label="posterior")
        plt.legend()
        # plt.title(f"{(e_bins[i] + e_bins[i+1])/2:.1f}, {(eta_bins[j] + eta_bins[j+1])/2:.1f}")
plt.show()


# print(percentiles, valid_percentiles)


print("Difference in posteior 95% and prior 95%:\n", 100*(percentiles - valid_percentiles)/valid_percentiles)

# Set the figure size and create the figure
fig = plt.figure(figsize=(10, 8))

# Create a colormap plot
ax = fig.add_subplot(111)
im = ax.imshow(quantiles.T, origin='lower', cmap='winter', aspect='auto', extent=[np.min(es), np.max(es), np.min(etas), np.max(etas)])
ax.set_xlabel(r'$e_0$', fontsize=16)
ax.set_ylabel(r'$\eta$', fontsize=16)
ax.set_title('Earth term search', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add text annotations to the colormap plot
for i in range(num_bins):
    for j in range(num_bins):
        value = quantiles[i, j]
        valid_value = valid_quantiles[i, j]
        if (value - valid_value)/valid_value > 0.05:
            ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{value:.2f}\n({valid_value:.2f})', fontsize=14,
                    color='black', ha='center', va='center')
        else:
            ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{value:.2f}\n({valid_value:.2f})', fontsize=14,
                    color='red', ha='center', va='center')


# Create the colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(r'95% upper limit on log$_{10}S_0$', fontsize=14)

plt.yticks(fontsize=14)

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Save the figure with desired size and aspect ratio
plt.savefig('Figures/earth_term_upperlim_lim_S0.pdf', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()

# # Create a colormap plot
# plt.figure(figsize=(8, 5))
# plt.imshow(percentiles.T, origin='lower', cmap='hot', aspect='auto', extent=[np.min(es), np.max(es), np.min(etas), np.max(etas)])
# plt.xlabel('First Parameter')
# plt.ylabel('Second Parameter')
# plt.title('95th Percentile of Third Parameter')
# plt.colorbar(label='Percentile')
# plt.show()


