import numpy as np
import matplotlib.pyplot as plt
import corner
import json

workdir = "simulated_partim"
true_params = json.load(open(f"{workdir}/true_gwecc_params.dat", "r"))

param_names = ['gwecc_cos_inc', 'gwecc_e0', 'gwecc_eta', 'gwecc_gamma0', 'gwecc_l0', 'gwecc_log10_A', 'gwecc_psi']
ndim = len(param_names)

truths = []
for i in range(ndim):
    truths.append(true_params[param_names[i][len("gwecc_") :]])


chain_file = "chains/chain_1.0.txt"
#chain_file = "chains2/chain_1.0.txt"
#chain_file = "chains_true/chain_1.0.txt"

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
    param_name = param_names[i][len("gwecc_") :]
    plt.axhline(truths[i], c="k")
    plt.ylabel(param_name)
plt.show()

# plt.figure(figsize=(10,30))
for i in range(ndim):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[burn:, 20+i])
    param_name = param_names[i][len("gwecc_") :]
    plt.axhline(truths[i], c="k")
    plt.ylabel(param_name)
plt.show()

plt.plot(chain[burn:, 20+ndim+1])
plt.ylabel('log_likelihood')
plt.title(f'last log_likelihood = {chain[-1, 20+ndim+1]}')
plt.show()

corner.corner(chain[burn:,20:-4], labels=param_names, truths=truths)
# plt.savefig("gwecc_sims/plots/corner.pdf")
plt.show()
