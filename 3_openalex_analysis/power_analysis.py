"""
This script performs power analysis for statistical tests using the TTestIndPower module from statsmodels library.
It calculates the required sample size to achieve a desired power level and significance level.
The script also generates power analysis plots to visualize the relationship between effect size, sample size, and power.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


# %% md
# the smaller the effect size, the harder it is to detect it with some kind
# of certainty, thus the larger is the required sample size for the statistical
# test.
# %%

effect_size = 0.07 # overlap/distance between means of the distributions (difference size)
alpha = 0.01 # significance level
power = 0.95 # 95% prob. that the test correctly rejects the null hypothesis (small p-value)

pwa = TTestIndPower()
sample_size = pwa.solve_power(effect_size = effect_size,
                             power = power,
                             alpha = alpha)
print(f"Required sample size: {sample_size:.2f}")


# %%
# Test run with synthetic data from example
alpha = 0.02 # significance level
sample_size_l = [100,150,200,250,300,400,600]
effect_size_l = np.linspace(0.01, 0.8, 51)

sns.set(style='whitegrid', font_scale=1.2, font='Arial')
fig, ax = plt.subplots(figsize=(6, 5))

TTestIndPower().plot_power(dep_var='es',
                            nobs=np.array(sample_size_l),
                            effect_size=np.linspace(0.01, 0.8, 51),
                            alpha=0.05,
                            ax=ax, title=r'$\alpha = 0.05$')
ax.set_xlabel("Effect Size")
ax.set_ylabel("Power")
ax.legend(frameon=False)
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.show()


# %%
savefig = False
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=1.1)
fig, ax = plt.subplots(figsize=(5, 4))
cls = sns.cubehelix_palette(start=.1, rot=.4, n_colors=len(sample_size_l))

n=0
for sample_sz in sample_size_l:
    # two-sided test default
    power_vals = (pwa.solve_power(effect_size=i,
                                nobs1=sample_sz,
                                alpha=alpha,
                                alternative='two-sided') for i in effect_size_l)
    ax.plot(effect_size_l, list(power_vals), label=sample_sz, color='salmon', lw=2)
    n+=1
ax.xaxis.set_major_locator(MaxNLocator(10))
ax.yaxis.set_major_locator(MaxNLocator(11))
ax.set_title(r"$\alpha$: " +f"{alpha}")
ax.set_xlabel("Effect Size")
ax.set_ylabel("Power")
ax.legend(title="Sample size", frameon=False)
plt.tight_layout()
sns.despine(left=True, bottom=True)
if savefig:
    plt.savefig("plots/power_analysis_curves.pdf" ,dpi=900)
plt.show()


# %% Make plot used in manuscript
alpha = 0.01  # significance level
sample_size_l = [500,1000,1500,2000,3000,4000]
sample_size_l = [round(sample_size)]
effect_size_l = np.linspace(0.01, 0.25, 51)

sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=1.1)
fig, ax = plt.subplots(figsize=(5, 4))
cls = sns.cubehelix_palette(start=.1, rot=.4, n_colors=len(sample_size_l))

n=0
for sample_sz in sample_size_l:
    # two-sided test default
    power_vals = (pwa.solve_power(effect_size=i,
                                nobs1=sample_sz,
                                alpha=alpha,
                                alternative='two-sided') for i in effect_size_l)
    ax.plot(effect_size_l, list(power_vals), label=sample_sz, color='salmon', lw=2)
    n+=1
ax.xaxis.set_major_locator(MaxNLocator(10))
ax.yaxis.set_major_locator(MaxNLocator(11))
ax.axvline(x=0.07, color='silver', lw=1.5, ls='--')
ax.set_title(r"Significance level $\alpha$: " +f"{alpha}")
ax.set_xlabel("Effect Size")
ax.set_ylabel("Power")
ax.legend(title="Sample size", frameon=False)
plt.tight_layout()
sns.despine(left=True, bottom=True)
if savefig:
    plt.savefig("plots/power_analysis_curve_draft.pdf" ,dpi=900)
plt.show()


# %% md
# # Considerations for calculationg out sample size
# Assuming we want to aim for high-power and low size-effect, we are designing
# our experiments based on values from the top-left quadrant of the plot.
# > Keywords freq needed N=100: 1429
# > Keywords freq needed N=150: 2143
# > Keywords freq needed N=200: 2857
# > Keywords freq needed N=250: 3571
# > Keywords freq needed N=300: 4286
# > Keywords freq needed N=400: 5714
# > Keywords freq needed N=600: 8571


# %% Print responses needed
rt = 0.089
for s in sample_size_l:
    print(f"Keywords freq needed N={s}: {s/rt:.0f}")

print(f"Final keywords freq needed N={sample_size:.0f}: {s/rt:.0f}")
