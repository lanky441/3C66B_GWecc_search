import numpy as np
import matplotlib.pyplot as plt
import corner
import json

workdir = "simulated_partim"
true_params = json.load(open(f"{workdir}/true_gwecc_params.dat", "r"))

param_names = ['gwecc_gwb_gamma', 'gwecc_gwb_log10A']
ndim = len(param_names)

truths = [4.333, np.log10(2.4e-15)]

chain_file = "chains_crn/chain_1.txt"

chain = np.loadtxt(chain_file)
print(chain.shape)

burn = chain.shape[0] // 4

for i in range(ndim):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[:, i])
    param_name = param_names[i][len("gwecc_") :]
    plt.axhline(truths[i], c="k")
    plt.ylabel(param_name)
plt.show()

# plt.figure(figsize=(10,30))
for i in range(ndim):
    plt.subplot(ndim, 1, i + 1)
    plt.plot(chain[burn:, i])
    param_name = param_names[i][len("gwecc_") :]
    plt.axhline(truths[i], c="k")
    plt.ylabel(param_name)
plt.show()

plt.plot(chain[burn:, ndim+1])
plt.ylabel('log_likelihood')
plt.title(f'last log_likelihood = {chain[-1, ndim+1]}')
plt.show()

corner.corner(chain[burn:,:-4], labels=param_names, truths=truths)
# plt.savefig("gwecc_sims/plots/corner.pdf")
plt.show()
