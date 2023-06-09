import numpy as np

def get_ew_groups(pta, name='gwecc'):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    params = pta.param_names
    ndim = len(params)
    # groups = [list(np.arange(0, ndim))]
    groups = []

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    print(f"snames = {snames}")
    
    if 'red noise' in snames:

        # create parameter groups for the red noise parameters
        rnpsrs = [p.split('_')[0] for p in params if 'red_noise_log10_A' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])

    if f'{name}_e0' in params:
        gpars = [x for x in params if name in x] #global params
        groups.append([params.index(gp) for gp in gpars]) #add global params

        #pair global params
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_eta')]])
        # groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_e0')]])
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_cos_inc')]])
        # groups.extend([[params.index(f'{name}_eta'), params.index(f'{name}_cos_inc')]])
        # groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_l0')]])
        groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_psi')]])
        # groups.extend([[params.index(f'{name}_psi'), params.index(f'{name}_l0')]])

    if 'gwb_gamma' in params:
        groups.extend([[params.index('gwb_gamma'), params.index('gwb_log10_A')]])
    
    psrdist_params = [ p for p in params if 'psrdist' in p ]
    lp_params = [ p for p in params if 'lp' in p ]
    gammap_params = [ p for p in params if 'gammap' in p ]
    
    if len(psrdist_params) !=0:
        for pd, lp, gp in zip (psrdist_params, lp_params, gammap_params):
            groups.extend([[params.index(pd), params.index(lp), params.index(gp), 
                                    params.index(f'{name}_log10_A')]])
    else:
        print("Not searching for psrdist!")
        
    return groups