import numpy as np
import pickle
from enterprise_extensions.empirical_distr import EmpiricalDistribution2D

chain = np.genfromtxt('NG12p5_chain.txt')
params = np.genfromtxt('NG12p5_params.txt', dtype='str')

burn = 30000

emp_dists = []

print(f"Chain shape = {chain.shape}")
nparams = len(params)
print(f"Number of params = {nparams}")
num_dists = int(nparams/2)

for i in range(num_dists):
    ndist = i+1
    samples = chain[burn:,2*i:2*ndist]
    param_names = params[2*i:2*ndist]
    print(param_names)
    
    gamma_bins = np.linspace(0, 7, 25)
    log10_A_bins = np.linspace(-20, -11, 25)
    
    emp_dists.append(EmpiricalDistribution2D(param_names, samples, 
                                             bins=[gamma_bins, log10_A_bins]))

with open('empirical_distributions.pkl', 'wb') as emp_dist_file:
    pickle.dump(emp_dists, emp_dist_file)