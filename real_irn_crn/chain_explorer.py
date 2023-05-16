import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_file", default="chains/chain_1.0.txt")

args = parser.parse_args()
chain_file = args.chain_file


param_names = ['gwb_gamma', 'gwb_log10_A', 'cos_inc', 'e0', 'eta', 'gamma0', 'l0', 'log10_A', 'psi']
ndim = len(param_names)

chain = np.loadtxt(chain_file)
print(chain.shape)

burn = chain.shape[0] //3

for i in range(20):
    plt.subplot(20, 1, i + 1)
    plt.plot(chain[:, i])
plt.show()

for i in range(20):
    plt.subplot(20, 1, i + 1)
    plt.plot(chain[burn:, i])
plt.show()

for i in range(ndim):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[:, 20+i])
    plt.ylabel(param_names[i])
plt.show()

# plt.figure(figsize=(10,30))
for i in range(ndim):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[burn:, 20+i])
    plt.ylabel(param_names[i])
plt.show()

plt.plot(chain[burn:, 20+ndim+1])
plt.ylabel('log_likelihood')
plt.title(f'last log_likelihood = {chain[-1, 20+ndim+1]}')
plt.show()

corner.corner(chain[burn:,20:-4], labels=param_names)
# plt.savefig("gwecc_sims/plots/corner.pdf")
plt.show()
