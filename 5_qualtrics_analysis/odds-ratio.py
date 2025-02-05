# %% codecell
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy as sp
import numpy as np
from collections import Counter
# import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import math
# import all data preprocessing steps for (consistency) across analysis
from preprocessing_results import \
df, cat_scores, agg_scores, cred_scores, agg_cred, egroup_l


# set uo working environment
prjwd = "../data"
savefig = False
plots_path = "plots_odds"

sns.set_theme(
    style="ticks",
    rc={'grid.linewidth': 0.5},
    font='Arial',
    font_scale=0.9)
cmap = sns.color_palette('PuOr', 4)


# %% Define help functions
def null_model(n, x):
    sample_l = []

    for i in range(n):
        # sampling from all responses to account for the whole system
        # taking 33% since groups are balanced
        sample = x.sample(frac=.33, random_state=rng)
        sample_count = sorted(Counter(sample).items())
        sample_l.append([i[1] for i in sample_count])
    print(*sample_count)
    return np.array(sample_l)


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


def sort_counts(x, prob=False):
    sorted_tuple = sorted(Counter(x).items())
    values = [i[1] for i in sorted_tuple]
    if prob:
        return [i/sum(values) for i in values]
    else:
        return values


def norm_counts(x):
    return [i/sum(x) for i in x]


# %% define estimaded gender df
df_cred = df[~df['credibility_num'].isna()]
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]
df_cred_est = df_cred[df_cred['estimated_gender'].ne("Uncertain/Don't know")]
rng = np.random.default_rng(seed=240220241138)

# %% md
# # Null models as defined in draft
# %%

null_var1 = 'credibility_num'
null_credibility = null_model(9999, df[~df[null_var1].isna()][null_var1])
display(
    null_var1,
    null_credibility.mean(axis=0),
    null_credibility.std(axis=0),
    np.median(null_credibility, axis=0)
    )

# %%
null_var2 = 'cite_likelihood_num'
null_cite_likelihood = null_model(9999, df[~df[null_var2].isna()][null_var2])
display(
    null_var2,
    null_cite_likelihood.mean(axis=0),
    null_cite_likelihood.std(axis=0),
    np.median(null_cite_likelihood, axis=0)
    )

# %%
null_var3 = 'estimated_gender'
null_estimated_gender = null_model(9999, df[~df[null_var3].isna()][null_var3])
display(
    null_var3,
    null_estimated_gender.mean(axis=0),
    null_estimated_gender.std(axis=0),
    np.median(null_estimated_gender, axis=0)
    )

# %%
null_credibility_est = null_model(9999, df_est[~df_est[null_var1].isna()][null_var1])
display(
    null_var1,
    null_credibility_est.mean(axis=0),
    null_credibility_est.std(axis=0),
    np.median(null_credibility_est, axis=0)
    )

# %%
null_cite_likelihood_est = null_model(9999, df_est[~df_est[null_var2].isna()][null_var2])
display(
    null_var2,
    null_cite_likelihood_est.mean(axis=0),
    null_cite_likelihood_est.std(axis=0),
    np.median(null_cite_likelihood_est, axis=0)
    )


# %% md
# # H1 & H2: Information theory hypothesis testing <> Null model
# ## Credibility & Cite likelihood
# %%
x_cat = [str(i) for i in cat_scores.values()]
x_cred = [str(i) for i in cred_scores.values()]
mks_l = ['o', '^', 'd', '*']
lw = 2


# %%
def plot_stem(variable, x, dx, null, title):
    fig, axs = plt.subplots(2, 1, figsize=(5, 5), tight_layout=True)
    markers1, stemlines1, baseline1 = axs[0].stem(
        x,
        np.array(sort_counts(dx[dx.egroup.eq('TFemale')][variable], True)) /
        np.array(sort_counts(dx[dx.egroup.eq('Control')][variable], True)),
        markerfmt='o', bottom=1, label='F/C', basefmt='none')
    markers2, stemlines2, baseline2 = axs[0].stem(
        x,
        np.array(sort_counts(dx[dx.egroup.eq('TMale')][variable], True)) /
        np.array(sort_counts(dx[dx.egroup.eq('Control')][variable], True)),
        markerfmt='s', bottom=1, label='M/C')
    markers3, stemlines3, baseline3 = axs[1].stem(
        x,
        np.array(sort_counts(dx[dx.egroup.eq('TFemale')][variable], True)) /
        norm_counts(null),
        markerfmt='o', bottom=1, label='F/Null', basefmt='none')
    markers4, stemlines4, baseline4 = axs[1].stem(
        x,
        np.array(sort_counts(dx[dx.egroup.eq('TMale')][variable], True)) /
        norm_counts(null),
        markerfmt='s', bottom=1, label='M/Null')
    # Adjust lines
    plt.setp([markers1, markers3], color=cmap[-1])
    plt.setp([markers2, markers4], color=cmap[0])
    plt.setp([stemlines1, stemlines3], color=cmap[-1])
    plt.setp([stemlines2, stemlines4], color=cmap[0])
    plt.setp([baseline2, baseline4], color='silver', ls='--', lw=1.5)
    # set labels
    axs[0].set_ylabel('Odds ratio - Control baseline')
    axs[1].set_ylabel('Odds ratio - Null model baseline')
    axs[1].set_xlabel(title)
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    sns.despine()
    if savefig:
        plt.savefig(f"{plots_path}/odds_{variable}_{title.replace(variable, '')}.pdf", dpi=300)
    plt.show()


# %%
plot_stem(
    'cite_likelihood_num', x_cat, df, null_cite_likelihood.mean(axis=0),
    'cite_likelihood_num')

# %%
plot_stem(
    'credibility_num', x_cred, df_cred, null_credibility.mean(axis=0),
    'credibility_num')


# %% md
# # H5 & H6: Information theory hypothesis testing <> Null model
# ## Credibility & Cite likelihood

# %%
plot_stem(
    'cite_likelihood_num', x_cat, df_est, null_cite_likelihood_est.mean(axis=0),
    'cite_likelihood_num - estimated only')

# %%
plot_stem(
    'credibility_num', x_cred, df_cred_est, null_credibility_est.mean(axis=0),
    'credibility_num - estimated gender only')
