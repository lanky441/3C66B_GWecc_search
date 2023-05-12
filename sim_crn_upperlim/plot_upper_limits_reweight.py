import matplotlib.pyplot as plt
import numpy as np
import sys
import wquantiles as wq
import nestle

def upper_limit_weight(value, pmin, pmax):
    """Reweight detection run into upper limit run."""
    mask = np.logical_and(value>=pmin, value<=pmax)
    uniform_prior = 1 / (pmax-pmin)
    linexp_prior = np.log(10) * 10**value / (10**pmax - 10**pmin)
    
    weight = mask * linexp_prior / uniform_prior
    weight /= sum(weight)

    return weight 

def plot_upper_limit(
    param_names,
    chain,
    xparam_name,
    xparam_label,
    xparam_lims,
    ampl_prior_0=1/5,
    xparam_bins=16,
    quantile=0.95,
    ylabel=False,
):
    ampl_idx = param_names.index("gwecc_log10_A")
    freq_idx = param_names.index(xparam_name)

    chain_ampl = chain[:, ampl_idx]
    chain_freq = chain[:, freq_idx]

    weights = upper_limit_weight(chain_ampl, -10, -5)

    log10_F_min, log10_F_max = xparam_lims
    log10_F_lins = np.linspace(log10_F_min, log10_F_max, xparam_bins + 1)
    log10_F_mins = log10_F_lins[:-1]
    log10_F_maxs = log10_F_lins[1:]
    log10_F_mids = (log10_F_mins + log10_F_maxs) / 2

    sd_bfs = []
    sd_bf_errs = []

    A_quant_Fbin = []
    A_violin_data = []
    for log10_fmin, log10_fmax in zip(log10_F_mins, log10_F_maxs):
        freq_mask = np.logical_and(chain_freq >= log10_fmin, chain_freq < log10_fmax)
        log10_A_Fbin_data = chain_ampl[freq_mask]
        weights_Fbin = weights[freq_mask]
        weights_Fbin /= sum(weights_Fbin)
        A_quant = wq.quantile(log10_A_Fbin_data, weights_Fbin, quantile)
        A_quant_Fbin.append(A_quant)
        A_violin_data_samples = 10**nestle.resample_equal(log10_A_Fbin_data, weights_Fbin) * 1e6
        A_violin_data.append(A_violin_data_samples)

        ampl_posterior_0 = np.histogram(log10_A_Fbin_data, density=True, bins=16)[0][0]
        sd_bf = ampl_prior_0 / ampl_posterior_0
        sd_bf_err = sd_bf / np.sqrt(len(log10_A_Fbin_data))
        sd_bfs.append(sd_bf)
        sd_bf_errs.append(sd_bf_err)

    overall_sd_bf = ampl_prior_0 / np.histogram(chain_ampl, density=True, bins=16)[0][0]
    overall_sd_bf_err = overall_sd_bf / np.sqrt(len(chain_ampl))

    A_quant = wq.quantile(chain_ampl, weights, quantile)

    print(np.transpose([log10_F_mins, log10_F_maxs, A_quant_Fbin]))

    violin_width = 3*(log10_F_max - log10_F_min) / (4*xparam_bins)

    plt.subplot(211)
    plt.violinplot(A_violin_data, positions=log10_F_mids, widths=violin_width, showextrema=False, quantiles=[[quantile]]*len(A_violin_data))
    plt.xlabel(xparam_label, fontsize=13)
    if ylabel:
        plt.ylabel("$A$ ($\mu$s)", fontsize=13)
    plt.xlim((log10_F_min, log10_F_max))
    plt.tick_params(labelsize=11)

    plt.subplot(212)
    plt.errorbar(log10_F_mids, sd_bfs, sd_bf_errs, marker="o", ls="")
    plt.axhspan(overall_sd_bf-overall_sd_bf_err, overall_sd_bf+overall_sd_bf_err, alpha=0.4)
    plt.xlabel(xparam_label, fontsize=13)
    plt.xlim((log10_F_min, log10_F_max))
    plt.ylabel("Savage-Dickey BF")

if __name__ == "__main__":
    np.int = np.int64

    chain_file = sys.argv[1]
    params_file = sys.argv[2]

    chain = np.genfromtxt(chain_file)
    with open(params_file, "r") as sf:
        param_names = [l.strip() for l in sf.readlines()]

    burn = chain.shape[0] // 4
    burned_chain = chain[burn:, :-4]

    plot_upper_limit(
        param_names,
        burned_chain,
        "gwecc_e0",
        "$e_0$",
        (0.01, 0.8),
        xparam_bins=8,
        quantile=0.95,
        ylabel=True,
    )

    plt.show()