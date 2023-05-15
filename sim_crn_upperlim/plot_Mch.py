import numpy as np
import astropy.units as u
import astropy.constants as c
import nestle
import matplotlib.pyplot as plt
import sys

from plot_upper_limits_reweight import upper_limit_weight
from enterprise_gwecc import jl

def plot_Mch(
    param_names,
    chain,
    gwdist,
    log10_F,
    reweight=True,
):
    def get_Mch(log10_A, e0, eta):
        Mch = jl.mass_from_gwdist(log10_A, log10_F, e0, gwdist, eta).Mch
        return (u.Quantity(Mch, "s") / (c.GM_sun/c.c**3)).to("").value

    burn = chain.shape[0] // 4
    burned_chain = chain[burn:, :-4]

    log10_As = burned_chain[:, param_names.index("gwecc_log10_A")]
    e0s = burned_chain[:, param_names.index("gwecc_e0")]
    etas = burned_chain[:, param_names.index("gwecc_eta")]

    Mchs = np.array([
        get_Mch(log10_A, e0, eta) 
        for log10_A, e0, eta in zip(log10_As, e0s, etas)
    ])
    if not reweight:
        Mchs = np.log10(Mchs)

    weights = upper_limit_weight(log10_As, -10, -5) if reweight else np.ones_like(log10_As)/len(log10_As)

    Mchs_reweighted = nestle.resample_equal(Mchs, weights)
    Mch_95up = np.quantile(Mchs_reweighted, 0.95)

    xparam_bins = 16
    e0_min, e0_max = 0.001, 0.8
    e0_lins = np.linspace(e0_min, e0_max, xparam_bins + 1)
    e0_mins = e0_lins[:-1]
    e0_maxs = e0_lins[1:]
    e0_mids = (e0_mins + e0_maxs) / 2

    Mch_violin_data = []
    for e0min, e0max in zip(e0_mins, e0_maxs):
        e0_mask = np.logical_and(e0s >= e0min, e0s < e0max)
        # log10_A_e0bin_data = log10_As[e0_mask]
        # e0_e0bin_data = e0s[e0_mask]
        # eta_e0bin_data = etas[e0_mask]

        weights_e0bin = weights[e0_mask]
        weights_e0bin /= sum(weights_e0bin)

        # Mch_samples_e0bin = np.array([
        #     get_Mch(log10_A, e0, eta) 
        #     for log10_A, e0, eta in zip(log10_A_e0bin_data, e0_e0bin_data, eta_e0bin_data)
        # ])
        Mch_samples_e0bin = Mchs[e0_mask]
        Mch_samples_e0bin_reweghted = nestle.resample_equal(Mch_samples_e0bin, weights_e0bin)
        Mch_violin_data.append(Mch_samples_e0bin_reweghted)

    violin_width = 3*(e0_max - e0_min) / (4*xparam_bins)

    plt.subplot(211)
    plt.violinplot(Mch_violin_data, positions=e0_mids, widths=violin_width, showextrema=False, quantiles=[[0.95]]*len(Mch_violin_data))
    Mch_label = "$M_{ch}$ ($M_{sun}$)" if reweight else "$\\log_{10} M_{ch}$ ($M_{sun}$)"
    plt.ylabel(Mch_label, fontsize=13)
    plt.xlabel("$e_0$", fontsize=13)
    plt.tick_params(labelsize=11)

    plt.subplot(212)
    plt.hist(Mchs_reweighted, density=True, bins=16)
    plt.axvline(Mch_95up)
    plt.xlabel(Mch_label, fontsize=13)

if __name__ == "__main__":
    np.int = np.int64

    chain_file = sys.argv[1]
    params_file = sys.argv[2]

    chain = np.genfromtxt(chain_file)
    with open(params_file, "r") as sf:
        param_names = [l.strip() for l in sf.readlines()]

    gwdist = 9.31
    log10_F = -7.219263270491185

    plot_Mch(
        param_names,
        chain,
        gwdist,
        log10_F,
        reweight=False,
    )
    plt.show()