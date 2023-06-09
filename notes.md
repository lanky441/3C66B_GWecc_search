# Dependencies
numpy
enterprise-pulsar
enterprise_extensions
GWecc.jl
PTMCMCSampler
juliacall

# search_irn_crn_gwecc.py

## Command line arguments

| Option         | Description                                             |
|----------------|---------------------------------------------------------|
| -s, --setting  | Settings file (default is `irn_crn_gwecc_psrterm.json`) | 

## Settings file
A JSON file containing
- datadir (Data directory, str)
- target_params (Target GW source parameters file, JSON, str)
- psrdist_info (Pulsar distance file, JSON)
- empirical_distr (Empirical distributions file, pickle, str)
- noise_dict (Noise dictionary file, JSON, str)
- psr_exclude (Pulsar black list, list<str>)
- psr_include (Pulsar white list; supersedes the black list, list<str>)
- gamma_vary (Vary the GWB index, bool)
- name (GWecc signal name, str)
- psrterm (Whether to include pulsar term in GWecc signal, bool)
- tie_psrterm (Whether to tie pulsar term phase of the GWecc signal to the Earth term phase (if psrterm), bool)
- x0_median (Whether to start sampling from the median values of previous runs, bool)
- Niter (Number of PTMCMC iterations, int)
- chaindir (Directory to save the PTMCMC chains, str)
- write_hotchain (Whether to write PTMCMC hot chains, bool)
- resume (Whether to resume PTMCMC run, bool)
- make_groups (??, bool)
- add_jumps (??, bool)
  
