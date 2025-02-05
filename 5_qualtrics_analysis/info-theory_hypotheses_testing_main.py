from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import textwrap
import scipy as sp
import numpy as np
from collections import Counter
# import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator, FixedLocator
from matplotlib.transforms import Affine2D
from statsmodels.miscmodels.ordinal_model import OrderedModel
import math
from scipy.stats import mannwhitneyu
# import all data preprocessing steps for (consistency) across analysis
from preprocessing_results import \
df, raw, cat_scores, agg_scores, cred_scores, agg_cred, egroup_l


# set uo working environment
prjwd = "../data"
savefig = False
plots_path = "plots_prob"
pd.options.display.float_format = '{:,.4f}'.format

sns.set_theme(
    style="ticks",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'y'},
    font='Arial',
    font_scale=0.9)

cl_eg = {
    'TFemale': "#d6604d",
    'Control': "#d9d9d9",
    'TMale': "#00c88a",
        }
cl_f, cl_c, cl_m = ["#d6604d", "#d9d9d9", "#00c88a"]
# %% Define help functions

def null_model(n, x):
    sample_l = []

    for i in range(n):
        # sampling from all responses to account for the whole system
        # taking 33% since groups are balanced
        # sample = x.sample(frac=.33, random_state=rng)
        sample = x.sample(frac=1, replace=True, random_state=rng).dropna()
        sample_count = sorted(Counter(sample).items())
        sample_l.append([i[1] for i in sample_count])
    print(*sample_count)
    sample_l = np.array(sample_l)
    norm_sample_l = sample_l / sample_l.sum(axis=1)[:, None]
    return norm_sample_l


def bootstrap(x, Nboot, statfunc):
    x = x.to_numpy()

    resampled_stat = []
    for k in range(Nboot):
        index = np.random.randint(0, len(x), len(x))
        sample = x[index]
        if statfunc == 'mode':
            b_statistic = sp.stats.mode(sample).mode
        else:
            b_statistic = statfunc(sample)
        resampled_stat.append(b_statistic)

    return np.array(resampled_stat)


def sort_counts(x, prob=False, show=False):
    sorted_tuple = sorted(Counter(x).items())
    values = [i[1] for i in sorted_tuple]
    if show:
        display(sorted_tuple)
    if prob is True:
        return [i/sum(values) for i in values]
    else:
        return values


def wrap_labels(ax, width, axis, break_long_words=False):
    "function to wrap y_label in many lines"
    labels = []
    if axis == 'y':
        ticklabels = ax.get_yticklabels()
    else:
        ticklabels = ax.get_xticklabels()
    for label in ticklabels:
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words, expand_tabs=True))
    if axis == 'y':
        ax.set_yticklabels(labels)
    else:
        ax.set_xticklabels(labels)


def cap(txt):
    if txt.startswith('eg'):
        return "Treatment group"
    else:
        return txt.replace('_', ' ').replace('-',' ').capitalize()


# %% md
# # Define auxiliary dataframes
# %%
df_cred = df[~df['credibility_num'].isna()]
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]
df_cred_est = df_cred[df_cred['estimated_gender'].ne("Uncertain/Don't know")]

# %% markdown
# # Check Countes and proportions
#  %% -------------------------------------------------------------------------
print(
    df[df.egroup.eq('Control')]['cite_likelihood'].count(),
    df[df.egroup.eq('TMale')]['cite_likelihood'].count(),
    df[df.egroup.eq('TFemale')]['cite_likelihood'].count())
print(df.cite_likelihood.count())

# %% Check counts of responses and totals
cl = Counter(df.cite_likelihood)
cl_g = df.groupby(['egroup'], observed=True)['cite_likelihood'].count()
display(cl, cl_g)
tot_cl = sum(cl.values())

cc = Counter(df[~df['credibility'].isna()]['credibility'])
cc_g = df.groupby(['egroup'], observed=True)['credibility'].count()
display(cc, cc_g)
tot_cc = sum(cc.values())
print(f'Total responses cite_likelihood: {tot_cl}')
print(f'Total responses credibility: {tot_cc}')

# %% md
# # Check the relative standard error (RSE)
# https://bookdown.org/Rmadillo/likert/how-many-respondents-are-enough.html
# %%
rse_cl = 1 / (math.sqrt(tot_cl)) * 100
rse_cc = 1 / (math.sqrt(tot_cc)) * 100
print(
    f"Relative standard error {'cite_likelihood' : <24} {rse_cl: .2f}%\n",
    f"Relative standard error {'credibility' : <24} {rse_cc: .2f}%",
    sep='')

for n, i in cl_g.items():
    rse = 1 / (math.sqrt(i)) * 100
    print(f"Relative standard error {'cite_likelihood' : <15} {n:<8} {rse: .2f}%")

for n, i in cc_g.items():
    rse = 1 / (math.sqrt(i)) * 100
    print(f"Relative standard error {'credibility' : <15} {n:<8} {rse: .2f}%")

# The standard error is very low for each group, but unclear how much this
# measure tells


# %% md
# # Effect sizes & CIs with bootstrapping
# %%
rng = np.random.default_rng(seed=240220241138)
run_boot = False

if run_boot:
    for g in egroup_l:
        cl_l = sorted(df[df.egroup.eq(g)]['cite_likelihood_num'].tolist())
        boot = sp.stats.bootstrap(
            (cl_l, ),
            np.mean,
            method='BCa',
            n_resamples=9999,
            random_state=rng)

        boot2 = sp.stats.bootstrap(
            (cl_l, ),
            np.median,
            method='basic',
            n_resamples=9999,
            random_state=rng)

        print(
            f"{g: <7}",
            f"{'mean': <7}",
            f"{boot.standard_error: .4f}",
            boot.confidence_interval,
        )
        print(
            f"{g: <7}",
            f"{'median': <7}",
            f"{boot2.standard_error: .4f}",
            boot2.confidence_interval,
        )
    boot2

# %%
if run_boot:
    for g in egroup_l:
        cl_l = sorted(df_cred[df_cred.egroup.eq(g)]['credibility_num'].tolist())
        boot = sp.stats.bootstrap(
            (cl_l, ),
            np.mean,
            method='BCa',
            n_resamples=9999,
            random_state=rng)

        boot2 = sp.stats.bootstrap(
            (cl_l, ),
            np.median,
            method='basic',
            n_resamples=9999,
            random_state=rng)

        print(
            f"{g: <7}",
            f"{'mean': <7}",
            f"{boot.standard_error: .4f}",
            boot.confidence_interval,
        )
        print(
            f"{g: <7}",
            f"{'median': <7}",
            f"{boot2.standard_error: .4f}",
            boot2.confidence_interval,
        )

# %% md
# # Null models as defined in draft
# %%

null_var = 'credibility_num'
null_credibility = null_model(9999, df[null_var])
display(
    null_var,
    null_credibility.mean(axis=0),
    null_credibility.std(axis=0),
    np.median(null_credibility, axis=0)
    )

# %%
null_var = 'cite_likelihood_num'
null_cite_likelihood = null_model(9999, df[null_var])
display(
    null_var,
    null_cite_likelihood.mean(axis=0),
    null_cite_likelihood.std(axis=0),
    np.median(null_cite_likelihood, axis=0)
    )

# %%
null_var = 'estimated_gender'
null_estimated_gender = null_model(9999, df[null_var])
display(
    null_var,
    null_estimated_gender.mean(axis=0),
    null_estimated_gender.std(axis=0),
    np.median(null_estimated_gender, axis=0)
    )

# %% md
# # Bootstrapping tests
# %%
plot_boot = False

if plot_boot:
    v = 'credibility_num'
    boot_md = bootstrap(df[~df[v].isna()][v], 9999, np.median)
    boot_me = bootstrap(df[~df[v].isna()][v], 9999, np.mean)
    boot_mo = bootstrap(df[~df[v].isna()][v], 9999, 'mode')

    'abc'.replace('ab', '')

    # Plot stats
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), tight_layout=True)
    axs[0].hist(boot_md)
    axs[1].hist(boot_me)
    axs[2].hist(boot_mo)
    axs[0].set_xlabel('Median')
    axs[1].set_xlabel('Mean')
    axs[2].set_xlabel('Mode')
    fig.suptitle(f'Bootstrapping 9999 epochs: {v}')
    sns.despine()
    if savefig:
        plt.savefig(f'{plots_path}/boot_{v}_distributions.pdf', dpi=300)
    plt.show()

# %%
if plot_boot:
    v = 'cite_likelihood_num'
    boot_md = bootstrap(df[~df[v].isna()][v], 9999, np.median)
    boot_me = bootstrap(df[~df[v].isna()][v], 9999, np.mean)
    boot_mo = bootstrap(df[~df[v].isna()][v], 9999, 'mode')

    # Plot stats
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), tight_layout=True)
    axs[0].hist(boot_md)
    axs[1].hist(boot_me)
    axs[2].hist(boot_mo)
    axs[0].set_xlabel('Median')
    axs[1].set_xlabel('Mean')
    axs[2].set_xlabel('Mode')
    fig.suptitle(f'Bootstrapping 9999 epochs: {v}')
    sns.despine()
    if savefig:
        plt.savefig(f'{plots_path}/boot_{v}_distributions.pdf', dpi=300)
    plt.show()


# %%
def calc_it_test(df, var, Qk, K):
    '''
    Calculate the relative entropy of given distributions.
    This routine will normalize pk and qk if they don't sum to 1.
    Kullback motivated the statistic as an expected log likelihood ratio.
    AICc: the smaller the value the “closer” to full reality and is a good
    approximation for the information in the data, relative to the other models
    Parasm:
    df: dataframe conaining data (non generic)
    var: variable
    Qk_count: reference distribution to compare with (e.g. null or control)
    K: Number of estimated parameters in the model (variables + the intercept)
    '''
    df = df[~df[var].isna()]
    labels_l = sorted(set(df.egroup))
    temp = []
    for label in labels_l:
        Pk = sort_counts(df[df.egroup == label][var])
        # log base 2
        KL_D = sp.stats.entropy(Pk, Qk, base=2)
        # CE = sp.stats.entropy(Pk, base=2) + KL_D
        n = len(Qk)
        logL = -(n/2) * np.log2(KL_D/n)
        AIC = -2 * logL + 2 * K
        AICc = AIC + (2*K*(K+1)) / (n-K-1)
        temp.append([label, KL_D, logL, AICc])
    # compute hypothesis suporting measures
    df_it = pd.DataFrame(temp, columns=['label', 'KL_D', 'logL', 'AICc'])
    df_it['deltai'] = df_it['AICc'] - df_it['AICc'].min()
    df_it['wi'] = np.exp((-1/2)*df_it['deltai'])
    df_it['wi'] = df_it['wi'] / df_it['wi'].sum()
    # return df_it.sort_values(by='deltai')
    df_it.sort_values(by='deltai', inplace=True, ignore_index=True)
    return df_it


# %%
def calc_odds_ci(counts_group1, counts_group2):
    # Hypothetical sample size for each group
    n1, n2 = sum(counts_group1), sum(counts_group2)
    # Calculate odds ratios and confidence intervals for each point on the Likert scale
    odds_ratios = []
    ci_lower = []
    ci_upper = []
    for i in range(len(counts_group1)):
        # Calculate odds for each point
        odds1 = counts_group1[i] / (n1 - counts_group1[i])
        odds2 = counts_group2[i] / (n2 - counts_group2[i])
        # Calculate odds ratio
        or_value = odds1 / odds2
        odds_ratios.append(or_value)
        # Calculate standard error for log(odds ratio)
        se_log_or = np.sqrt(1/counts_group1[i] + 1/(n1 - counts_group1[i]) + 1/counts_group2[i] + 1/(n2 - counts_group2[i]))
        # Confidence interval for log(odds ratio)
        ci_log_or = [np.log(or_value) - 1.96 * se_log_or, np.log(or_value) + 1.96 * se_log_or]
        # Convert back to odds ratio scale
        ci_or = np.exp(ci_log_or)
        ci_lower.append(ci_or[0])
        ci_upper.append(ci_or[1])
    return np.array(odds_ratios), np.array(ci_upper), np.array(ci_lower)

def calc_odds(counts_group1, counts_group2):
    # Hypothetical sample size for each group
    n1, n2 = sum(counts_group1), sum(counts_group2)
    # Calculate odds ratios and confidence intervals for each point on the Likert scale
    odds_ratios = []
    for i in range(len(counts_group1)):
        # Calculate odds for each point
        odds1 = counts_group1[i] / (n1 - counts_group1[i])
        odds2 = counts_group2[i] / (n2 - counts_group2[i])
        # Calculate odds ratio
        or_value = odds1 / odds2
        odds_ratios.append(or_value)
    return np.array(odds_ratios)


def calc_odds_ci_boot(group1, group2, nboot, rng, show_boot=False):
    res_boot = []
    for i in range(nboot):
        # get same size sample with replacement
        sample_g1 = group1.sample(frac=1, replace=True, random_state=rng)
        sample_g2 = group2.sample(frac=1, replace=True, random_state=rng)
        # get ordered counts/probabilities
        counts_group1 = np.array(sort_counts(sample_g1))
        counts_group2 = np.array(sort_counts(sample_g2))

        odds_ratios = calc_odds(counts_group1, counts_group2)
        res_boot.append(odds_ratios)

    empirical_odds_ratios = calc_odds(
        np.array(sort_counts(group1)),
        np.array(sort_counts(group2))
        )
    res_boot = np.stack(res_boot)
    ci_low, ci_up = np.percentile(res_boot, [2.5, 97.5], axis=0)
    if show_boot:
        return empirical_odds_ratios, ci_low, ci_up, res_boot
    else:
        return empirical_odds_ratios, ci_low, ci_up


def calc_ttest(dx, var, pair, nboot, rng):
    i,j = pair
    odds, ci_lo, ci_up, boot = calc_odds_ci_boot(
        dx[dx.egroup.eq(i)][var],
        dx[dx.egroup.eq(j)][var],
        nboot, rng, True)
    # test each likert point vector individually for h0 = 1.0 (equal odds)
    res_tttest = []
    for point in range(boot.shape[1]):
        obj = sp.stats.ttest_1samp(boot[point], popmean=1.0, alternative='two-sided')
        res_tttest.append(obj.pvalue)
    return res_tttest, odds, ci_lo, ci_up


# %% md
# # Run t-test for each odds ratio
# %%
run_test = False
save_df = False

if run_test:
    pairs = {
        ('TFemale', 'Control'): 'F/C',
        ('TMale', 'Control'): 'M/C',
        ('TFemale', 'TMale'): 'F/M'
    }
    nboot = 9999
    result_ttest = []

    # All responses only
    for pair, label in pairs.items():
        # compute bootstrap and get odds ratio vector
        var = 'cite_likelihood_num'
        ttest, odds, ci_lo, ci_up = calc_ttest(df, var, pair, nboot, rng)
        result_ttest.append(['df', pair, var, list(zip(ttest, odds, ci_lo, ci_up))])

        var = 'credibility_num'
        ttest, odds, ci_lo, ci_up = calc_ttest(df_cred, var, pair, nboot, rng)
        result_ttest.append(['df_cred', pair, var, list(zip(ttest, odds, ci_lo, ci_up))])


    # Estimated gender responses only
    for pair in pairs:
        # compute bootstrap and get odds ratio vector
        var = 'cite_likelihood_num'
        ttest, odds, ci_lo, ci_up = calc_ttest(df_est, var, pair, nboot, rng)
        result_ttest.append(['df_est', pair, var, list(zip(ttest, odds, ci_lo, ci_up))])

        var = 'credibility_num'
        ttest, odds, ci_lo, ci_up = calc_ttest(df_cred_est, var, pair, nboot, rng)
        result_ttest.append(['df_cred_est', pair, var, list(zip(ttest, odds, ci_lo, ci_up))])


    # build dataframe
    temp_ttest = []
    for i in result_ttest:
        if i[2].startswith('cite'):
            score = list(cat_scores.values())
        else:
            score = list(cred_scores.values())
        n=0
        for tt, od, lo, up in i[3]:
            temp_ttest.append([i[0], pairs[i[1]], i[1], i[2], score[n], tt, od, lo, up])
            n+=1
    cols = ['dataset', 'label', 'pair', 'variable', 'score', 'ttest_pval', 'odds', 'ci_lo', 'ci_up']
    df_tt = pd.DataFrame(temp_ttest, columns=cols)

    if save_df:
        df_tt.to_csv('odds_ttest.csv', index=False)


# %% md
# # Information theory hypothesis testing <> Control
# # Control as a reference doesn't make much sense
# ## Credibility & Cite likelihood

# %% Qk_count includes all responses, use all data as reference distribution
variable = 'cite_likelihood_num'
Qk_count = sort_counts(df[df.egroup == 'Control'][variable], True)
display(calc_it_test(df[df.egroup.ne('Control')], variable, Qk_count, 2))
print(f"Weight of evidence: {0.855494/0.144506:.1f} times")


# %% Qk_count includes all responses, use all data as reference distribution
variable = 'credibility_num'
# df_cred = df[~df[variable].isna()]
Qk_count = sort_counts(df_cred[df_cred['egroup'].eq('Control')][variable], True)
display(
    calc_it_test(df_cred[df_cred.egroup.ne('Control')], variable, Qk_count, 2))
print(f"Weight of evidence: {0.767307/0.232693:.1f} times")


# %% md
# # H1 & H2: Information theory hypothesis testing <> Null model
# ## Credibility & Cite likelihood

# %%
variable = 'cite_likelihood_num'
Qk_count = null_cite_likelihood.mean(axis=0)  # already sorted
Qk_std = null_cite_likelihood.std(axis=0)
# Qk_count = [i/sum(Qk_count) for i in Qk_count]
display(calc_it_test(df, variable, Qk_count, 2))
c = calc_it_test(df, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")
print(f"Weight of evidence: {c['wi'][2]/c['wi'][1]:.1f} times")

# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df[df.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)

axs[0].legend(loc='lower left')
odds, ci_up, ci_lo = calc_odds_ci(
    np.array(sort_counts(df[df.egroup.eq('TFemale')][variable])),
    np.array(sort_counts(df[df.egroup.eq('TMale')][variable]))
    )
axs[1].plot(
    x,
    odds,
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].fill_between(x, ci_lo, ci_up, alpha=0.15, color=fc)
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Cite likelihood')
axs[1].legend(loc='lower left')
axs[1].yaxis.set_major_locator(MaxNLocator(7))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}.pdf', dpi=300)
plt.show()


# %%
fig, axs = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df[df.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='upper left', frameon=False)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.5)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.66, 0.7, 0.32, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.axhline(y=1, color='lightgrey', lw=2, ls='--')
ax2.plot(
    x,
    np.array(sort_counts(df[df.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df[df.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
ax2.set_ylabel('Odds ratio')
ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='lower left', frameon=False)
ax2.yaxis.set_major_locator(MaxNLocator(4))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df[df.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.5)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.71, 0.7, 0.28, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
markers1, stemlines1, baseline1 = ax2.stem(
    x,
    np.array(sort_counts(df[df.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df[df.egroup.eq('Control')][variable], True)),
    markerfmt='o', bottom=1, label='F/C', basefmt='none')
markers2, stemlines2, baseline2 = ax2.stem(
    x,
    np.array(sort_counts(df[df.egroup.eq('TMale')][variable], True)) /
    np.array(sort_counts(df[df.egroup.eq('Control')][variable], True)),
    markerfmt='d', bottom=1, label='M/C')
# Adjust lines
plt.setp([markers1], color=cl_eg['TFemale'], markersize=5)
plt.setp([markers2], color=cl_eg['TMale'], markersize=5)
plt.setp([stemlines1], color=cl_eg['TFemale'])
plt.setp([stemlines2], color=cl_eg['TMale'])
plt.setp([baseline2], color='silver', ls='--', lw=1)
ax2.set_ylim(0.86, 1.21)
ax2.legend(loc='upper left', frameon=False, fontsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(7))

left3, bottom3, width3, height3 = [0.31, 0.7, 0.28, 0.3]
ax3 = fig.add_axes([left3, bottom3, width3, height3])
ax3.axhline(y=1, color='lightgrey', lw=2, ls='--')
odds, ci_up, ci_lo = calc_odds_ci(
    np.array(sort_counts(df[df.egroup.eq('TFemale')][variable])),
    np.array(sort_counts(df[df.egroup.eq('TMale')][variable]))
    )
ax3.plot(
    x,
    odds,
    label='F/M', marker='o', ms=5, c='steelblue')
ax3.fill_between(x, ci_lo, ci_up, alpha=0.15, color=fc)
ax3.set_ylabel('Odds ratio')
ax3.legend(loc='lower left', frameon=False, fontsize=8)
ax3.yaxis.set_major_locator(MaxNLocator(4))
# ax3.set_ylim(0.75, 1.07)
ax3.yaxis.set_major_locator(MaxNLocator(7))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset-both.pdf', dpi=300,
    bbox_inches='tight')
plt.show()


# %% load bootstraped CI from file
try:
    print(df_tt.shape)
except NameError:
    df_tt = pd.read_csv('odds_ttest.csv')

df_oc_f =  df_tt[df_tt.dataset.eq('df') & df_tt.variable.eq(variable) & df_tt.label.eq('F/C')]
odds_f, ci_lo_f, ci_up_f = df_oc_f['odds'].values, df_oc_f['ci_lo'].values, df_oc_f['ci_up'].values

df_oc_m =  df_tt[df_tt.dataset.eq('df') & df_tt.variable.eq(variable) & df_tt.label.eq('M/C')]
odds_m, ci_lo_m, ci_up_m = df_oc_m['odds'].values, df_oc_m['ci_lo'].values, df_oc_m['ci_up'].values

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'

axs.axhline(y=1, color='lightgrey', lw=2, ls='--')
trans1 = Affine2D().translate(-0.1, 0.0) + axs.transData
trans2 = Affine2D().translate(+0.1, 0.0) + axs.transData

axs.errorbar(
    x,
    odds_f,
    yerr=[odds_f-ci_lo_f, ci_up_f-odds_f],
    label='F/C', marker='o', ls='None', c=cl_eg['TFemale'], capsize=4, transform=trans1)

axs.errorbar(
    x,
    odds_m,
    yerr=[odds_m-ci_lo_m, ci_up_m-odds_m],
    label='M/C', marker='d', ls='None', c=cl_eg['TMale'], capsize=4, transform=trans2)
axs.set_xlabel('Cite likelihood')
axs.set_ylabel('Odds ratio')
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.yaxis.set_major_locator(MaxNLocator(10))

left, bottom, width, height = [0.35, 0.72, 0.25, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])
for n, g in enumerate(egroup_l):
    ax2.plot(
        x,
        sort_counts(df[df.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=4,
        c=cl_eg[g], lw=1.5)
ax2.plot(
    Qk_count,
    label='Null model', ls='--', color='k', lw=1)
ax2.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
ax2.legend(loc='lower right', frameon=False, fontsize=6, bbox_to_anchor=(1.55, .25))
ax2.set_ylabel('Probability', fontsize=10)
ax2.set_ylim(0.02, 0.31)
ax2.tick_params(axis='both', labelsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset_errorbars.pdf', dpi=300,
    bbox_inches='tight')
plt.show()


# %%
fig, ax = plt.subplots(figsize=(2.2, 3), constrained_layout=True)
sat=1
clrs_g = [cl_c, cl_f, cl_m]
estimator = 'median'

sns.boxplot(data=df_est,
            x="egroup", y="cite_likelihood_num", hue='egroup',
            width=0.6, palette=clrs_g, saturation=sat, ax=ax)\
            .set(
                title='All responses',
                ylabel='Cite likelihood',
                xlabel=None)
sns.despine(bottom=True, offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/boxplot_egroup_cite-likelihood_all.pdf', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.2, 3), constrained_layout=True)
error = 'sd'
# error = ('ci', 95)
estimator = 'median'
sns.pointplot(
    data=df_est,
    x="egroup", y="cite_likelihood_num", hue='egroup',
    estimator=estimator, errorbar=error, palette=clrs_g, ax=ax)\
    .set(xlabel='Estimated gender',
        ylabel=f'{estimator.capitalize()} cite likelihood',
        title='All responses')
# ax.set_ylim(-3, 3)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_{estimator}-cite-likelihood_control.pdf', dpi=300)
plt.show()

# %% --------------------------------------------------------------------------
variable = 'credibility_num'
Qk_count = null_credibility.mean(axis=0)  # already sorted
Qk_std = null_credibility.std(axis=0)
display(calc_it_test(df_cred, variable, Qk_count, 2))
c = calc_it_test(df_cred, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")


# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs[0].legend()
axs[1].plot(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Credibility')
axs[1].legend()
axs[1].yaxis.set_major_locator(MaxNLocator(5))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='upper left', frameon=False)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.8)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.69, 0.68, 0.28, 0.28]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.axhline(y=1, color='lightgrey', lw=2, ls='--')
ax2.plot(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
ax2.set_ylabel('Odds ratio')
ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='upper left', frameon=False)
ax2.yaxis.set_major_locator(MaxNLocator(4))

sns.despine(offset=.05, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='upper left', frameon=False)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.8)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.65, 0.68, 0.33, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
markers1, stemlines1, baseline1 = ax2.stem(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('Control')][variable], True)),
    markerfmt='o', bottom=1, label='F/C', basefmt='none')
markers2, stemlines2, baseline2 = ax2.stem(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TMale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('Control')][variable], True)),
    markerfmt='d', bottom=1, label='M/C')
# Adjust lines
plt.setp([markers1], color=cl_eg['TFemale'], markersize=5)
plt.setp([markers2], color=cl_eg['TMale'], markersize=5)
plt.setp([stemlines1], color=cl_eg['TFemale'])
plt.setp([stemlines2], color=cl_eg['TMale'])
plt.setp([baseline2], color='silver', ls='--', lw=1)
ax2.legend(loc='upper left', frameon=False, fontsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(5))

sns.despine(offset=.05, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset-lolipop.pdf', dpi=300)
plt.show()

# %%
df_oc_f =  df_tt[df_tt.dataset.eq('df_cred') & df_tt.variable.eq(variable) & df_tt.label.eq('F/C')]
odds_f, ci_lo_f, ci_up_f = df_oc_f['odds'].values, df_oc_f['ci_lo'].values, df_oc_f['ci_up'].values

df_oc_m =  df_tt[df_tt.dataset.eq('df_cred') & df_tt.variable.eq(variable) & df_tt.label.eq('M/C')]
odds_m, ci_lo_m, ci_up_m = df_oc_m['odds'].values, df_oc_m['ci_lo'].values, df_oc_m['ci_up'].values

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'

axs.axhline(y=1, color='lightgrey', lw=2, ls='--')
trans1 = Affine2D().translate(-0.1, 0.0) + axs.transData
trans2 = Affine2D().translate(+0.1, 0.0) + axs.transData

axs.errorbar(
    x,
    odds_f,
    yerr=[odds_f-ci_lo_f, ci_up_f-odds_f],
    label='F/C', marker='o', ls='None', c=cl_eg['TFemale'], capsize=4, transform=trans1)

axs.errorbar(
    x,
    odds_m,
    yerr=[odds_m-ci_lo_m, ci_up_m-odds_m],
    label='M/C', marker='d', ls='None', c=cl_eg['TMale'], capsize=4, transform=trans2)
axs.set_xlabel('Credibility')
axs.set_ylabel('Odds ratio')
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.set_ylim(0.7, 1.5)
axs.yaxis.set_major_locator(MaxNLocator(9))

left, bottom, width, height = [0.36, 0.74, 0.3, 0.22]
ax2 = fig.add_axes([left, bottom, width, height])
for n, g in enumerate(egroup_l):
    ax2.plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=4,
        c=cl_eg[g], lw=1.5)
ax2.plot(
    Qk_count,
    label='Null model', ls='--', color='k', lw=1)
ax2.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
ax2.legend(loc='lower right', frameon=False, fontsize=6, bbox_to_anchor=(1.4, .25))
ax2.set_ylabel('Probability', fontsize=10)
# ax2.set_ylim(0.02, 0.31)
ax2.tick_params(axis='both', labelsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset_errorbars.pdf', dpi=300,
    bbox_inches='tight')
plt.show()


# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.set_ylabel('Probability')
axs.set_xlabel('Credibility')
axs.set_ylim(0, 0.8)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.72, 0.72, 0.28, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
markers1, stemlines1, baseline1 = ax2.stem(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('Control')][variable], True)),
    markerfmt='o', bottom=1, label='F/C', basefmt='none')
markers2, stemlines2, baseline2 = ax2.stem(
    x,
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TMale')][variable], True)) /
    np.array(sort_counts(df_cred[df_cred.egroup.eq('Control')][variable], True)),
    markerfmt='d', bottom=1, label='M/C')
# Adjust lines
plt.setp([markers1], color=cl_eg['TFemale'], markersize=5)
plt.setp([markers2], color=cl_eg['TMale'], markersize=5)
plt.setp([stemlines1], color=cl_eg['TFemale'])
plt.setp([stemlines2], color=cl_eg['TMale'])
plt.setp([baseline2], color='silver', ls='--', lw=1)
# ax2.set_ylabel('Odds ratio')
# ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='upper left', frameon=False, fontsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(5))

left3, bottom3, width3, height3 = [0.32, 0.72, 0.28, 0.3]
ax3 = fig.add_axes([left3, bottom3, width3, height3])
ax3.axhline(y=1, color='lightgrey', lw=2, ls='--')
odds, ci_up, ci_lo = calc_odds_ci(
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TFemale')][variable])),
    np.array(sort_counts(df_cred[df_cred.egroup.eq('TMale')][variable]))
    )
ax3.plot(
    x,
    odds,
    label='F/M', marker='o', ms=5, c='steelblue')
ax3.fill_between(x, ci_lo, ci_up, alpha=0.15, color=fc)
ax3.set_ylabel('Odds ratio')
# ax3.set_xlabel('Cite likelihood')
ax3.legend(loc='upper left', frameon=False, fontsize=8)
ax3.set_ylim(0.81, 1.38)
ax3.yaxis.set_major_locator(MaxNLocator(6))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_inset-both.pdf', dpi=300,
    bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.2, 3), constrained_layout=True)
sat=1
clrs_g = [cl_c, cl_f, cl_m]
hue_order = [3,2,1,-1,-2,-3]
sns.boxplot(data=df_cred,
            x="egroup", y="credibility_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=ax)\
            .set(
                title='All responses',
                ylabel='Credibility',
                xlabel=None)
ax.yaxis.set_major_locator(MaxNLocator(5))
sns.despine(bottom=True, offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/boxplot_egroup_credibility_all.pdf', dpi=300)
plt.show()

# %% md
# # H5 & H6: Information theory hypothesis testing <> Null model
# ## Credibility & Cite likelihood for estimated gender
# %% define estimaded gender df
# df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]
# df_cred_est = df_cred[df_cred['estimated_gender'].ne("Uncertain/Don't know")]
print(df_est.shape[0]/df.shape[0])

# %% define null models
variable = 'cite_likelihood_num'
Qk_count = null_cite_likelihood.mean(axis=0)  # already sorted
Qk_std = null_cite_likelihood.std(axis=0)
display(calc_it_test(df_est, variable, Qk_count, 2))
c = calc_it_test(df_est, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")


# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cat_scores.values()]
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df_est[df_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs[0].legend(loc='lower left')
axs[1].plot(
    x,
    np.array(sort_counts(df_est[df_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_est[df_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Cite likelihood')
axs[1].legend()
sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_est[df_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='upper left', frameon=False)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.5)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.69, 0.7, 0.3, 0.26]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.axhline(y=1, color='lightgrey', lw=2, ls='--')
ax2.plot(
    x,
    np.array(sort_counts(df_est[df_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_est[df_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
ax2.set_ylabel('Odds ratio')
ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='lower left', frameon=False)
ax2.yaxis.set_major_locator(MaxNLocator(5))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null_inset.pdf', dpi=300)
plt.show()


# %%
df_oc_f =  df_tt[df_tt.dataset.eq('df_est') & df_tt.variable.eq(variable) & df_tt.label.eq('F/C')]
odds_f, ci_lo_f, ci_up_f = df_oc_f['odds'].values, df_oc_f['ci_lo'].values, df_oc_f['ci_up'].values

df_oc_m =  df_tt[df_tt.dataset.eq('df_est') & df_tt.variable.eq(variable) & df_tt.label.eq('M/C')]
odds_m, ci_lo_m, ci_up_m = df_oc_m['odds'].values, df_oc_m['ci_lo'].values, df_oc_m['ci_up'].values

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'

axs.axhline(y=1, color='lightgrey', lw=2, ls='--')
trans1 = Affine2D().translate(-0.1, 0.0) + axs.transData
trans2 = Affine2D().translate(+0.1, 0.0) + axs.transData
axs.errorbar(
    x,
    odds_f,
    yerr=[odds_f-ci_lo_f, ci_up_f-odds_f],
    label='F/C', marker='o', ls='None', c=cl_eg['TFemale'], capsize=4, transform=trans1)
axs.errorbar(
    x,
    odds_m,
    yerr=[odds_m-ci_lo_m, ci_up_m-odds_m],
    label='M/C', marker='d', ls='None', c=cl_eg['TMale'], capsize=4, transform=trans2)
axs.set_xlabel('Cite likelihood')
axs.set_ylabel('Odds ratio')
axs.legend(loc='upper right', frameon=False, bbox_to_anchor=(.9, 1))
axs.set_ylim(0.3, 1.5)
axs.yaxis.set_major_locator(MaxNLocator(12))

left, bottom, width, height = [0.37, 0.27, 0.23, 0.19]
ax2 = fig.add_axes([left, bottom, width, height])
for n, g in enumerate(egroup_l):
    ax2.plot(
        x,
        sort_counts(df_est[df_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=4,
        c=cl_eg[g], lw=1.5)
ax2.plot(
    Qk_count,
    label='Null model', ls='--', color='k', lw=1)
ax2.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
ax2.legend(loc='lower right', frameon=False, fontsize=6, bbox_to_anchor=(1.75, .0))
ax2.set_ylabel('Probability', fontsize=10)
# ax2.set_ylim(0.02, 0.31)
ax2.tick_params(axis='both', labelsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est_inset_errorbars.pdf', dpi=300,
    bbox_inches='tight')
plt.show()


# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cat_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_est[df_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.5)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.71, 0.7, 0.28, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
markers1, stemlines1, baseline1 = ax2.stem(
    x,
    np.array(sort_counts(df_est[df_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_est[df_est.egroup.eq('Control')][variable], True)),
    markerfmt='o', bottom=1, label='F/C', basefmt='none')
markers2, stemlines2, baseline2 = ax2.stem(
    x,
    np.array(sort_counts(df_est[df_est.egroup.eq('TMale')][variable], True)) /
    np.array(sort_counts(df_est[df_est.egroup.eq('Control')][variable], True)),
    markerfmt='d', bottom=1, label='M/C')
# Adjust lines
plt.setp([markers1], color=cl_eg['TFemale'], markersize=5)
plt.setp([markers2], color=cl_eg['TMale'], markersize=5)
plt.setp([stemlines1], color=cl_eg['TFemale'])
plt.setp([stemlines2], color=cl_eg['TMale'])
plt.setp([baseline2], color='silver', ls='--', lw=1)
# ax2.set_ylabel('Odds ratio')
# ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='lower left', frameon=False, fontsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

left3, bottom3, width3, height3 = [0.31, 0.7, 0.28, 0.3]
ax3 = fig.add_axes([left3, bottom3, width3, height3])
ax3.axhline(y=1, color='lightgrey', lw=2, ls='--')
odds, ci_up, ci_lo = calc_odds_ci(
    np.array(sort_counts(df_est[df_est.egroup.eq('TFemale')][variable])),
    np.array(sort_counts(df_est[df_est.egroup.eq('TMale')][variable]))
    )
ax3.plot(
    x,
    odds,
    label='F/M', marker='o', ms=5, c='steelblue')
ax3.fill_between(x, ci_lo, ci_up, alpha=0.15, color=fc)
ax3.set_ylabel('Odds ratio')
# ax3.set_xlabel('Cite likelihood')
ax3.legend(loc='lower left', frameon=False, fontsize=8)
ax3.yaxis.set_major_locator(MaxNLocator(5))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null_inset-both.pdf', dpi=300,
    bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.2, 3), constrained_layout=True)
sat=1
clrs_g = [cl_c, cl_f, cl_m]
hue_order = [3,2,1,-1,-2,-3]

sns.boxplot(data=df_est,
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=ax)\
            .set(
                title='Estimated gender responses',
                ylabel='Cite likelihood',
                xlabel=None)
sns.despine(bottom=True, offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/boxplot_egroup_cite-likelihood_est-gender.pdf', dpi=300)
plt.show()


# %% --------------------------------------------------------------------------
print(df_cred_est.shape[0]/df_cred.shape[0])
variable = 'credibility_num'
Qk_count = null_credibility.mean(axis=0)  # already sorted
Qk_std = null_credibility.std(axis=0)
display(calc_it_test(df_cred_est, variable, Qk_count, 2))
c = calc_it_test(df_cred_est, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")

# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df_cred_est[df_cred_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs[0].legend()
axs[1].plot(
    x,
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Credibility')
axs[1].legend()
plt.legend()

sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred_est[df_cred_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='upper left', frameon=False)
axs.set_ylabel('Probability')
axs.set_xlabel('Cite likelihood')
axs.set_ylim(0, 0.8)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.69, 0.68, 0.28, 0.28]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.axhline(y=1, color='lightgrey', lw=2, ls='--')
ax2.plot(
    x,
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
ax2.set_ylabel('Odds ratio')
ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='upper left', frameon=False)
ax2.yaxis.set_major_locator(MaxNLocator(4))

sns.despine(offset=.05, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null_inset.pdf', dpi=300)
plt.show()

# %%
df_oc_f =  df_tt[df_tt.dataset.eq('df_cred_est') & df_tt.variable.eq(variable) & df_tt.label.eq('F/C')]
odds_f, ci_lo_f, ci_up_f = df_oc_f['odds'].values, df_oc_f['ci_lo'].values, df_oc_f['ci_up'].values

df_oc_m =  df_tt[df_tt.dataset.eq('df_cred_est') & df_tt.variable.eq(variable) & df_tt.label.eq('M/C')]
odds_m, ci_lo_m, ci_up_m = df_oc_m['odds'].values, df_oc_m['ci_lo'].values, df_oc_m['ci_up'].values

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'

axs.axhline(y=1, color='lightgrey', lw=2, ls='--')
trans1 = Affine2D().translate(-0.1, 0.0) + axs.transData
trans2 = Affine2D().translate(+0.1, 0.0) + axs.transData

axs.errorbar(
    x,
    odds_f,
    yerr=[odds_f-ci_lo_f, ci_up_f-odds_f],
    label='F/C', marker='o', ls='None', c=cl_eg['TFemale'], capsize=4, transform=trans1)

axs.errorbar(
    x,
    odds_m,
    yerr=[odds_m-ci_lo_m, ci_up_m-odds_m],
    label='M/C', marker='d', ls='None', c=cl_eg['TMale'], capsize=4, transform=trans2)
axs.set_xlabel('Credibility')
axs.set_ylabel('Odds ratio')
axs.legend(loc='upper right', frameon=False, bbox_to_anchor=(.9, 1))
axs.set_ylim(0.3, 1.57)
axs.yaxis.set_major_locator(MaxNLocator(13))

left, bottom, width, height = [0.42, 0.27, 0.27, 0.19]
ax2 = fig.add_axes([left, bottom, width, height])
for n, g in enumerate(egroup_l):
    ax2.plot(
        x,
        sort_counts(df_cred_est[df_cred_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=4,
        c=cl_eg[g], lw=1.5)
ax2.plot(
    Qk_count,
    label='Null model', ls='--', color='k', lw=1)
ax2.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
ax2.legend(loc='lower right', frameon=False, fontsize=6, bbox_to_anchor=(1.65, 0))
ax2.set_ylabel('Probability', fontsize=10)
# ax2.set_ylim(0.02, 0.31)
ax2.tick_params(axis='both', labelsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est_inset_errorbars.pdf', dpi=300,
    bbox_inches='tight')
plt.show()

# %%
fig, axs = plt.subplots(figsize=(4, 2.8), constrained_layout=True)
x = [str(i) for i in cred_scores.values()]
mks_l = ['o', '^', 'd', '*']
fc = 'salmon'
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred_est[df_cred_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color=fc)
axs.legend(loc='lower left', frameon=False, ncols=2)
axs.set_ylabel('Probability')
axs.set_xlabel('Credibility')
axs.set_ylim(0, 0.8)
axs.yaxis.set_major_locator(MaxNLocator(6))

left, bottom, width, height = [0.72, 0.71, 0.28, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
markers1, stemlines1, baseline1 = ax2.stem(
    x,
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('Control')][variable], True)),
    markerfmt='o', bottom=1, label='F/C', basefmt='none')
markers2, stemlines2, baseline2 = ax2.stem(
    x,
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TMale')][variable], True)) /
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('Control')][variable], True)),
    markerfmt='d', bottom=1, label='M/C')
# Adjust lines
plt.setp([markers1], color=cl_eg['TFemale'], markersize=5)
plt.setp([markers2], color=cl_eg['TMale'], markersize=5)
plt.setp([stemlines1], color=cl_eg['TFemale'])
plt.setp([stemlines2], color=cl_eg['TMale'])
plt.setp([baseline2], color='silver', ls='--', lw=1)
# ax2.set_ylabel('Odds ratio')
# ax2.set_xlabel('Cite likelihood')
ax2.legend(loc='lower left', frameon=False, fontsize=8)
ax2.yaxis.set_major_locator(MaxNLocator(6))

left3, bottom3, width3, height3 = [0.32, 0.71, 0.28, 0.3]
ax3 = fig.add_axes([left3, bottom3, width3, height3])
ax3.axhline(y=1, color='lightgrey', lw=2, ls='--')
odds, ci_up, ci_lo = calc_odds_ci(
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TFemale')][variable])),
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TMale')][variable]))
    )
ax3.plot(
    x,
    odds,
    label='F/M', marker='o', ms=5, c='steelblue')
ax3.fill_between(x, ci_lo, ci_up, alpha=0.15, color=fc)
ax3.set_ylabel('Odds ratio')
# ax3.set_xlabel('Cite likelihood')
ax3.legend(loc='upper left', frameon=False, fontsize=8)
ax3.yaxis.set_major_locator(MaxNLocator(5))

sns.despine(offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null_inset-both.pdf', dpi=300,
    bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.2, 3), constrained_layout=True)
sat=1
clrs_g = [cl_c, cl_f, cl_m]
hue_order = [3,2,1,-1,-2,-3]

sns.boxplot(data=df_cred_est,
            x="egroup", y="credibility_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=ax)\
            .set(
                title='Estimated gender responses',
                ylabel='Credibility',
                xlabel=None)
ax.yaxis.set_major_locator(MaxNLocator(5))
sns.despine(bottom=True, offset=.1, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/boxplot_egroup_credibility_est-gender_null.pdf', dpi=300)
plt.show()


# %% md
# # H5 & H6: Information theory hypothesis testing <> Null model estimated gender only
# ## Credibility & Cite likelihood for estimated gender
# %%
null_var = 'credibility_num'
null_credibility_est = null_model(9999, df_est[~df_est[null_var].isna()][null_var])
display(
    null_var,
    null_credibility_est.mean(axis=0),
    null_credibility_est.std(axis=0),
    np.median(null_credibility_est, axis=0)
    )

# %%
null_var = 'cite_likelihood_num'
null_cite_likelihood_est = null_model(9999, df_est[~df_est[null_var].isna()][null_var])
display(
    null_var,
    null_cite_likelihood_est.mean(axis=0),
    null_cite_likelihood_est.std(axis=0),
    np.median(null_cite_likelihood_est, axis=0)
    )

# %%
variable = 'cite_likelihood_num'
Qk_count = null_cite_likelihood_est.mean(axis=0)  # already sorted
Qk_std = null_cite_likelihood_est.std(axis=0)
display(calc_it_test(df_est, variable, Qk_count, 2))
c = calc_it_test(df_est, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")

# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cat_scores.values()]
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df_est[df_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs[0].legend()
axs[1].plot(
    x,
    np.array(sort_counts(df_est[df_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_est[df_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Cite likelihood')
axs[1].legend()
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null-est.pdf', dpi=300)
plt.show()

# %%
variable = 'credibility_num'
Qk_count = null_credibility_est.mean(axis=0)  # already sorted
Qk_std = null_credibility_est.std(axis=0)

display(calc_it_test(df_cred_est, variable, Qk_count, 2))
c = calc_it_test(df_cred_est, variable, Qk_count, 2)
print(f"Weight of evidence: {c['wi'][0]/c['wi'][1]:.1f} times")
print(f"Weight of evidence: {c['wi'][0]/c['wi'][2]:.1f} times")


# %%
fig, axs = plt.subplots(2, 1, figsize=(4, 5), tight_layout=True, sharex=False)
x = [str(i) for i in cred_scores.values()]
for n, g in enumerate(egroup_l):
    axs[0].plot(
        x,
        sort_counts(df_cred_est[df_cred_est.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs[0].plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs[0].fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs[0].legend()
axs[1].plot(
    x,
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TFemale')][variable], True)) /
    np.array(sort_counts(df_cred_est[df_cred_est.egroup.eq('TMale')][variable], True)),
    label='F/M', marker='o', ms=6, c='steelblue')
axs[1].axhline(y=1, color='lightgrey', lw=2, ls='--')
axs[0].set_ylabel('Probability')
axs[1].set_ylabel('Odds ratio')
axs[1].set_xlabel('Credibility')
axs[1].legend()
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/probab_odds_{variable}_est-gender_null-est.pdf', dpi=300)
plt.show()


# %% md
# # H3: Print Count and proportions of estimated gender
# ## Hypothesis confirmed, 67.85% are uncertain\Don't know while 32,15%
# ## reported a gender
# %%
est_gender = df[~df['estimated_gender'].isna()]
est_gender_c = Counter(est_gender['estimated_gender'])
est_gender_c
est_gender_r = {i: j/est_gender_c.total() for i, j in est_gender_c.items()}
est_gender_r
est_gender.shape

print(
    *(f"{i:<24}{j: .2%}" for i, j in est_gender_r.items()),
    sep='\n'
    )

# %%
reasons_c = Counter(df[~df['estimated_gender_reason'].isna()]['estimated_gender_reason'])
reasons_r = {i: j/reasons_c.total() for i, j in reasons_c.items()}
reasons_r

print(
    *(f"{i:<47}{j: .2%}" for i, j in reasons_r.items()),
    sep='\n'
    )

print(df[~df['estimated_gender_reason'].isna()].shape[0]/est_gender.shape[0])

# %%
Qk_count = null_estimated_gender.mean(axis=0)
Qk_std = null_estimated_gender.std(axis=0)
variable = 'estimated_gender'
fig, axs = plt.subplots(figsize=(4, 3), tight_layout=True, sharex=False)
x = sorted([str(i) for i in est_gender_c.keys()])
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(est_gender[est_gender.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs.set_ylabel('Probability')
axs.set_xlabel('Estimated gender')
axs.legend(loc='best')
wrap_labels(axs, 10, 'x')
sns.despine(offset=.01, trim=True)

if savefig:
    plt.savefig(f'{plots_path}/probab{variable}_est-gender.pdf', dpi=300)
plt.show()

# %%
cmap3 = sns.color_palette(clrs_g)

sns.set_theme(style="white",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'y'},
    font='Arial',
    font_scale=0.9)

variable = "estimated_gender"
ax = sns.displot(
    data=df,
    x=variable,
    row='egroup',
    height=1.45,
    kde=True,
    aspect=2,
    color='slategrey',
    )

ax.set_xlabels(cap(variable))
ax.set_ylabels('Responses')
ax.tick_params(axis='x', labelrotation=45)
ax.set_titles('{row_name}', color='gray', y=.6, x=.15)
wrap_labels(ax.axes[2][0], 12, 'x')
plt.subplots_adjust(hspace=-1)
for z in ax.axes:
    z[0].yaxis.set_major_locator(MaxNLocator(4))

ax.fig.tight_layout()
sns.despine(left=True)
if savefig:
    plt.savefig(f'{plots_path}/hist{variable}_vert.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(.7,4.5))

clr_s = ['salmon','slategrey', 'slategrey','slategrey']
bottom = 0
n = 0
for category, value in est_gender_c.most_common():
    ax.bar("Responses", value, bottom=bottom, label=category,
        color=clr_s[n], alpha=.8)
    bottom += value
    n += 1

t1 = f"{list(est_gender_r.values())[0]:.1%}"
ax.annotate(t1+" Uncertain/Don't know", xy=(1.75, .33), xytext=(1.85, .12), xycoords='figure fraction',
    rotation=90, arrowprops=dict(arrowstyle='-[, widthB=7, lengthB=1', color="0.5",
                    connectionstyle='angle'))
t2 = f"{sum(list(est_gender_r.values())[1:]):.1%}"
ax.annotate(t2+' All others', xy=(1.75, .69), xytext=(1.85, .58), xycoords='figure fraction',
    rotation=90, arrowprops=dict(arrowstyle='-[, widthB=3.2, lengthB=1', color="0.6",
                    connectionstyle='angle'))

sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/bar_{variable}_vert.pdf', dpi=300, bbox_inches='tight')
plt.show()



# %% -----
variable = "estimated_gender_reason"
sorted_order = df[variable].value_counts().index
df[variable] = pd.Categorical(df[variable], categories=sorted_order, ordered=True)

# %%
sns.set_theme(style="whitegrid",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'x'},
    font='Arial',
    font_scale=0.9)

ax = sns.displot(
    data=df,
    y=variable,
    multiple='stack',
    hue='egroup',
    alpha=.8,
    label='Responses',
    height=3.7,
    aspect=1.7,
    palette=cmap3,
    legend=True)

ax.set_ylabels(cap(variable))
ax.set_xlabels('Frequency')
sns.move_legend(obj=ax, loc='lower right', bbox_to_anchor=(.83, 0.17),
    frameon=False, title='Group')

wrap_labels(ax.ax, 23, 'y')
if savefig:
    plt.savefig(f'{plots_path}/hist{variable}.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
raw_t = raw.copy()
raw_t.duration_secs = raw_t.duration_secs/60

raw_t[(~raw_t.estimated_gender_reason.isna()) & (raw_t.duration_secs <= 5)].shape[0]
raw_t[~raw_t.estimated_gender_reason.isna()].shape[0]
print(2919/4609)
# raw_t[raw_t.duration_secs > 30000].shape[0]

fig, axs = plt.subplots(figsize=(4, 3), tight_layout=True, sharex=False)
axs.hist(raw['duration_secs'])
axs.set_yscale('log')
plt.show()


# %% md
# # H4 Print Count and proportions of Credibility
# ## Hypothesis rejected, credibility score sums to -2761.0, indicating that
# ## in general, 63,11%, the abstract is Not credible or Somewhat credible
# %%
credibility_c = Counter(df[~df['credibility'].isna()]['credibility'])
credibility_c
credibility_c_r = {i: j/credibility_c.total() for i, j in credibility_c.items()}
credibility_c_r
print(
    *(f"{i:<24}{j: .2%}" for i, j in credibility_c_r.items()),
    sep='\n'
    )
credibility_cn = Counter(df[~df['credibility_num'].isna()]['credibility_num'])
display(credibility_cn)
credibility_sum = df[~df['credibility_num'].isna()]['credibility_num'].sum()
print(f"Total credibility score: {credibility_sum}")
credibility_s = Counter(df[~df['credibility_sign'].isna()]['credibility_sign'])
credibility_s_r = {i: j/credibility_s.total() for i, j in credibility_s.items()}

# %%
sns.set_theme(
    style="ticks",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'y'},
    font='Arial',
    font_scale=0.9)


variable = 'credibility_num'
Qk_count = null_credibility.mean(axis=0)  # already sorted
Qk_std = null_credibility.std(axis=0)  # already sorted
fig, axs = plt.subplots(figsize=(4, 3), tight_layout=True, sharex=False)
x = [str(i) for i in cred_scores.keys()]
for n, g in enumerate(egroup_l):
    axs.plot(
        x,
        sort_counts(df_cred[df_cred.egroup.eq(g)][variable], True),
        label=g,
        marker=mks_l[n],
        ms=6,
        c=cl_eg[g])
axs.plot(
    Qk_count,
    label='Null model', ls='--', color='k')
axs.fill_between(x, Qk_count-Qk_std, Qk_count+Qk_std, alpha=0.2, color='k')
axs.set_ylabel('Probability')
axs.set_xlabel('Credibility')
axs.legend(loc='best', ncol=2)
wrap_labels(axs, 10, 'x')
axs.yaxis.set_major_locator(MaxNLocator(7))
sns.despine(offset=.01, trim=True)
if savefig:
    plt.savefig(f'{plots_path}/probab{variable}_est-gender.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'x'},
    font='Arial',
    font_scale=0.9)
variable = "credibility"
ax = sns.displot(
    data=df,
    y=variable,
    multiple='stack',
    hue='egroup',
    alpha=.8,
    label='Responses',
    height=2.4,
    aspect=1.8,
    palette=cmap3)
ax.set_ylabels(cap(variable))
ax.set_xlabels('Frequency')
wrap_labels(ax.ax, 22, 'y')
if savefig:
    plt.savefig(f'{plots_path}/hist{variable}.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
sns.set_theme(style="white",
    rc={'grid.linewidth': 0.5, 'axes.grid.axis':'y'},
    font='Arial',
    font_scale=0.9)

ax = sns.displot(
    data=df,
    x=variable,
    row='egroup',
    height=1.4,
    kde=True,
    aspect=2,
    color='slategrey',
    )

ax.set_xlabels(cap(variable))
ax.set_ylabels('Responses')
ax.tick_params(axis='x', labelrotation=45)
ax.set_titles('{row_name}', color='gray', y=.6, x=.15)
wrap_labels(ax.axes[2][0], 11, 'x')
plt.subplots_adjust(hspace=-1)
for z in ax.axes:
    z[0].yaxis.set_major_locator(MaxNLocator(4))

ax.fig.tight_layout()
sns.despine(left=True)
if 1:
    plt.savefig(f'{plots_path}/hist{variable}_vert.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(.7,4.5))

clr_s = ['salmon','salmon', 'slategrey','slategrey']
bottom = 0
n = 0
ordered = [credibility_c.most_common()[i] for i in [2,0,1,3]]

for category, value in ordered:
    ax.bar("Responses", value, bottom=bottom, label=category,
        color=clr_s[n], alpha=.8)
    bottom += value
    n += 1
credibility_c_r
t1 = f"{credibility_c_r['Not credible']+credibility_c_r['Somewhat credible']:.1%}"
ax.annotate(t1+" Not Credible", xy=(1.75, .31), xytext=(1.9, .18), xycoords='figure fraction',
    rotation=90, fontsize=10, arrowprops=dict(arrowstyle='-[, widthB=7.2, lengthB=1', color="0.5",
                    connectionstyle='angle'))
t2 = f"{credibility_c_r['Quite credible']+credibility_c_r['Very credible']:.1%}"
ax.annotate(t2+' Credible', xy=(1.75, .68), xytext=(1.9, .58), xycoords='figure fraction',
    rotation=90, fontsize=10, arrowprops=dict(arrowstyle='-[, widthB=3.8, lengthB=1', color="0.6",
                    connectionstyle='angle'))

sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/bar_{variable}_vert.pdf', dpi=300, bbox_inches='tight')
plt.show()
