# %% codecell
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import Counter

prjwd = "../data"
savefig = False
plots_path = "plots"


# %% markdown
# # Clean responses and make some checks
# %% codecell
cat_scores = {
    'Strongly unlikely': -3,
    'Quite unlikely': -2,
    'Somewhat unlikely': -1,
    'Somewhat likely': 1,
    'Quite likely': 2,
    'Strongly likely': 3
    }

# %% codecell
fname = "sciofsci_double-effect_-_Live_April_15,_2024_07.14_anonymised.csv"
raw = pd.read_csv(os.path.join(prjwd, fname), header=0)

# Progress distribution
Counter(raw.Progress)
Counter(raw.Finished)
Counter(raw.seen_survey_before)
# >= 45: Have responded to at least the main question, filtering out
# those who seen the survey before and did not consent
# raw = raw[raw.Progress >= 45]


# --------------------------------------------------------------------------
# %% Create a smaller copy
df = raw[[
    'ResponseId', 'duration_secs', 'credibility',
    'estimated_gender', 'occupation', 'participant_gender',
    'TFemale', 'Control', 'TMale'
    ]].copy()

# melt treatment groups into single column
df_long = df[[
    'ResponseId', 'estimated_gender', 'TFemale', 'Control', 'TMale']].melt(
        id_vars=['ResponseId', 'estimated_gender'],
        var_name='egroup',
        value_name='cite_likelihood').reset_index(drop=True)
df_long

# %%
# drop empty rows since there's only 1 answear for each group
# TODO check the effect of keeping partial answears
df_long = df_long.dropna().copy()

# set datatypes for effieciency
df_long['cite_likelihood'] = df_long['cite_likelihood'].astype("category")
df_long['estimated_gender'] = df_long['estimated_gender'].str.split('/').str[0].astype("str")
df_long['egroup'] = df_long['egroup'].astype("category")


# Create numeric cite_likelihood column
df_long['cite_likelihood_num'] = df_long['cite_likelihood'].apply(
    lambda x: cat_scores[x])
df_long['cite_likelihood_num'] = df_long['cite_likelihood_num'].astype("int")

# Create binart sign of cite_likelihood
bins = [-4, 0, 3]
names = ['Unlikely', 'Likely']
df_long['sign'] = pd.cut(df_long['cite_likelihood_num'], bins, labels=names)
df_long['sign_num'] = df_long['cite_likelihood_num'].apply(
    lambda x: 1 if x > 0 else -1)

df_long.head(3)

# %% md
# # Some validations
# %%
Counter(df_long.cite_likelihood)
len(set(df_long.ResponseId))
df_long.dtypes
Counter(df_long.estimated_gender)
df_long[df_long.estimated_gender == "Man"]
# %% --------------------------------------------------------------------------
# define ploting params
sns.set_theme(
    style="ticks",
    rc={'grid.linewidth': 0.5},
    font='Arial',
    font_scale=0.9)
cl_f, cl_c, cl_m = ["#d6604d", "#d9d9d9", "#00c88a"]
hue_order = [3, 2, 1, -1, -2, -3]
# cmap = sns.color_palette('PuOr', 6)
cmap = 'PuOr'

# %% md
# # Final plots
# %%
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()

sns.histplot(data=df_long,
            x="egroup", hue='cite_likelihood_num', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax = axs[0])\
            .set(title='All responses', xlabel=None, ylabel='Proportion')
sns.histplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", hue='cite_likelihood_num', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[1])\
            .set(title='Male', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", hue='cite_likelihood_num', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[2])\
            .set(title='Female', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", hue='cite_likelihood_num', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax= axs[3])\
            .set(title="Uncertain/Don't know", xlabel=None)

sns.move_legend(axs[3], "upper right", bbox_to_anchor=(1.7, 1.04), frameon=False, title='Cite\n likelihood')
fig.suptitle('Estimated gender')
# fig.supxlabel('Experiment group')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_egroup_proportion.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]
cl_f, cl_c, cl_m = ["#d6604d", "#d9d9d9", "#00c88a"]
clrs_g = [cl_c, cl_f, cl_m]
sat = 1
sns.boxplot(data=df_long,
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=axs[0])\
            .set(title='All responses', ylabel='Cite likelihood', xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxtplot_egroup_mean-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(2.4, 3), constrained_layout=True)
# clrs_ge = ["#00c88a", "#f1a340", "#abd9e9"]
cl_u = "#abd9e9"
clrs_g = [cl_c, cl_f, cl_m]

sns.boxplot(data=df_long,
            x="egroup", y="cite_likelihood_num", width=0.6,
            palette=clrs_g, hue='egroup', saturation=sat, ax=ax)\
            .set(
                title='All responses',
                ylabel='Cite likelihood',
                xlabel=None)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxplot_egroup_cite-likelihood_all.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
# error = 'sd'
error = ('ci', 95)
estimator = 'mean'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelihood_num",
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[0], estimator=estimator)\
            .set(title='All responses', ylabel='Mean cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", y="cite_likelihood_num",
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[1], estimator=estimator)\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", y="cite_likelihood_num",
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[2], estimator=estimator)\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", y="cite_likelihood_num",
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[3], estimator=estimator)\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_mean-cite-likelihood.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelihood_num", estimator='median',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Median cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", y="cite_likelihood_num",  estimator='median',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", y="cite_likelihood_num",  estimator='median',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", y="cite_likelihood_num",  estimator='median',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_median-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelihood_num", estimator='mode',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Mode cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", y="cite_likelihood_num",  estimator='mode',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", y="cite_likelihood_num",  estimator='mode',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", y="cite_likelihood_num",  estimator='mode',
            palette=clrs_g, hue='egroup', errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_mode-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
df_long_ctrl = df_long[
    (df_long.egroup == 'Control') &
    (df_long.estimated_gender != 'Other')
    ].sort_values(by='estimated_gender')

xtlabels = ['Female', 'Male', 'Uncertain']
df_long_ctrl.groupby(
    by=['estimated_gender', 'cite_likelihood'],
    observed=False
    ).size()

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.histplot(data=df_long_ctrl,
            x="estimated_gender", hue='cite_likelihood_num', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax = ax)\
            .set(xlabel='Estimated gender',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.35, 1.04), frameon=False, title='Cite\n likelihood')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion_control.pdf', dpi=300)
plt.show()
# Counter(df_long_ctrl.estimated_gender)
# Counter(df_long.estimated_gender)

# %%
df_long_ctrl[df_long_ctrl.cite_likelihood_num > 0].groupby('estimated_gender').count()
df_long_ctrl[df_long_ctrl.sign == 1].groupby('estimated_gender')['cite_likelihood_num'].sum()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
# clrs_ge = ["#00c88a", "#f1a340", "#abd9e9"]
cl_u = "#abd9e9"
clrs_ge = [cl_f, cl_m, cl_u]

sns.boxplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelihood_num", width=0.6,
            palette=clrs_ge, hue="estimated_gender", saturation=sat, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Cite likelihood',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxplot_est-gender_mean-cite-likelihood_control.pdf', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelihood_num",
            palette=clrs_ge, hue="estimated_gender", errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Mean cite likelihood',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_mean-cite-likelihood_control.pdf', dpi=300)
plt.show()

# %%
df_long_ctrl.groupby('estimated_gender').size()
df_long.groupby('estimated_gender').size()

# %%
fig, ax = plt.subplots(figsize=(2.4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelihood_num", estimator='median',
            palette=clrs_ge, hue="estimated_gender", errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Median cite likelihood',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_median-cite-likelihood_control.pdf', dpi=300)
plt.show()

# %%
df_long_estd = df_long[df_long.estimated_gender!='Other']\
    .sort_values(by='estimated_gender')

df_long_estd[
    df_long_estd.egroup == 'Control'].groupby(
        by=['estimated_gender', 'cite_likelihood'],
        observed=False).size()


# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]

sns.histplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", hue='cite_likelihood_num', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[0])\
            .set(xlabel=None,
                title='TFemale group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", hue='cite_likelihood_num', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[1])\
            .set(xlabel=None,
                title='Control group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", hue='cite_likelihood_num', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax=axs[2])\
            .set(xlabel=None,
                title='TMale group')

# for a in axs: a.set_xticklabels(xtlabels)
sns.move_legend(axs[2], "upper right", bbox_to_anchor=(1.45, 1.04), frameon=False, title='Cite\n likelihood')
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)

sns.boxplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", y="cite_likelihood_num", width=0.6,
            palette=clrs_ge, hue='estimated_gender', saturation=sat, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Cite likelihood',
                title='TFemale group')
sns.boxplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="cite_likelihood_num", width=0.6,
            palette=clrs_ge, hue='estimated_gender', saturation=sat, ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.boxplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="cite_likelihood_num", width=0.6,
            palette=clrs_ge, hue='estimated_gender', saturation=sat, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

# for a in axs: a.set_xticklabels(xtlabels)
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxtplot_est-gender_proportion.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", y="cite_likelihood_num",
            palette=clrs_ge, hue='estimated_gender', errorbar=error, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Mean cite likelihood',
                title='TFemale group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="cite_likelihood_num",
            palette=clrs_ge, hue='estimated_gender', errorbar=error,ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="cite_likelihood_num",
            palette=clrs_ge, hue='estimated_gender', errorbar=error, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

# for a in axs: a.set_xticklabels(xtlabels)
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_proportion.pdf', dpi=300)
plt.show()

# %% md
# # Final plots aggregated

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]
alpha = 1
sns.histplot(data=df_long,
            x="sign", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax = axs[0])\
            .set(title='All responses', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Man"],
            x="sign", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax=axs[1])\
            .set(title='Male', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="sign", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax=axs[2])\
            .set(title='Female', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="sign", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, ax= axs[3])\
            .set(title="Uncertain/Don't know", xlabel=None)

sns.move_legend(axs[3], "upper right", bbox_to_anchor=(1.7, 1.04), frameon=False, title='Group')
fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_cite-likelihood-agg_proportion.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="sign_num",
            palette=clrs_g, hue="egroup", errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Mean cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Man"],
            x="egroup", y="sign_num",
            palette=clrs_g, hue="egroup", errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Woman"],
            x="egroup", y="sign_num",
            palette=clrs_g, hue="egroup", errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain"],
            x="egroup", y="sign_num",
            palette=clrs_g, hue="egroup", errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_mean-cite-likelihood-agg.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
hue_order = ['Likely', 'Unlikely']
# clrs_agg = ["#91accf", "#d19491"]
clrs_agg = [sns.color_palette(cmap).as_hex()[0], sns.color_palette(cmap).as_hex()[-1]]
alpha = 0.75

fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.histplot(data=df_long_estd,
            x="estimated_gender", hue='sign', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, alpha=alpha, ax=ax)\
            .set(xlabel='Estimated gender',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.4, 1.04), frameon=False, title='Cite\n likelihood')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion-agg_control.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="sign_num",
            palette=clrs_ge, hue="estimated_gender", errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Mean cite likelihood',
                title='Control group')
# ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_mean-cite-likelihood-agg_control.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()

sns.histplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", hue='sign', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, legend=False, alpha=alpha, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Proportion',
                title='TFemale group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", hue='sign', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, legend=False, alpha=alpha, ax=axs[1])\
            .set(xlabel=None,
                title='Control group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", hue='sign', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, alpha=alpha, ax=axs[2])\
            .set(xlabel=None,
                title='TMale group')

# for a in axs: a.set_xticklabels(xtlabels)
sns.move_legend(axs[2], "upper right", bbox_to_anchor=(1.57, 1.04), frameon=False, title='Cite\n likelihood')
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion-agg.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", y="sign_num",
            palette=clrs_ge, hue="estimated_gender", errorbar=error, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Mean cite likelihood',
                title='TFemale group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="sign_num",
            palette=clrs_ge, hue="estimated_gender", errorbar=error,ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="sign_num",
            palette=clrs_ge, hue="estimated_gender", errorbar=error, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

# for a in axs: a.set_xticklabels(xtlabels)
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_proportion-agg.pdf', dpi=300)
plt.show()


# %% md
# # Response rate per keywords
# ---
# %%
fname_contacts = ""  #file not included for provacy reasons
df_r = pd.read_csv(os.path.join(prjwd, fname_contacts), header=0)
df_r.head(3)
df_r.shape
inv_count = df_r[df_r.inviteCount > 0].groupby('keywords').size().reset_index(name='invites')
res_count = df_r[df_r.responseCount > 0].groupby('keywords').size().reset_index(name='responses')
df_rates = inv_count.merge(res_count, on='keywords')
df_rates['rate'] = df_rates['responses'] / df_rates['invites']
df_rates['keywords'] = df_rates['keywords'].apply(lambda x: x.replace('-',' ').replace('_',', '))
df_rates.head(3)
print(len(set(df_rates.keywords)))

# %% difference between sets
set(inv_count['keywords']) - set(res_count['keywords'])
len(set(df_r[df_r.inviteCount > 0]['ContactID']))
len(set(df_r[df_r.responseCount > 0]['ContactID']))
Counter(df_r.responseCount)
df_rates.rate.mean()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.85)
fig, axs = plt.subplots(1,3, figsize=(8,3), constrained_layout=True)
axs = axs.flatten()

sns.ecdfplot(data=df_rates,
            x="rate", complementary=True,
            ax=axs[0])\
            .set(xlabel='Response rate by keyword',
                ylabel='Complementary ECDF')
sns.histplot(data=df_rates,
            x="rate", stat='frequency',
            ax=axs[1])\
            .set(xlabel='Response rate by keyword',
                ylabel='Invites sent')
sns.scatterplot(data=df_rates,
            x="rate", y='invites',
            ax=axs[2])\
            .set(xlabel='Response rate by keyword',
                ylabel='Invites sent')

sns.despine()
# if savefig:
    # plt.savefig(f'{plots_path}/response-rate-by-keywords_ecdf_hist_scatter.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.75)
fig, ax = plt.subplots(figsize=(4.7, 6), constrained_layout=True)
top = round(df_rates.shape[0]*0.1)
dr1 = df_rates.sort_values(by='rate', ascending=False).iloc[:top]

sns.barplot(
    data=dr1,
    y="keywords", x='rate',
    color='#BA7A75',
    alpha=.7,
    label='Rate', ax=ax)\
    .set(xlabel='Response rate', ylabel=None)

ax.tick_params(axis='y', labelsize=8, labelcolor='slategrey')
for bar, label in zip(ax.patches, dr1['invites']):
    y = bar.get_y()
    width = bar.get_width()
    height = bar.get_height()
    ax.text(width+0.015, y+height, label, ha="center", fontsize=7, color='dimgrey')
ax.axvline(x=df_rates.rate.mean(), color='lightgrey', lw=1.5, ls='--')
ax.text(
    df_rates.rate.mean()+.12, 35,
    f"Mean: {df_rates.rate.mean(): .2%}",
    ha="left", fontsize=9, color='dimgrey')
ax.set_title('Top 10% keywords by response rate')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/response-rate_keywords_bar.pdf', dpi=300)
plt.show()


# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.75)
fig, ax = plt.subplots(figsize=(4.7, 6), constrained_layout=True)
top = round(df_rates.shape[0]*0.1)
by = 'responses'
dr2 = df_rates.sort_values(by=by, ascending=False).iloc[:top]

sns.barplot(
    data=dr2,
    y="keywords", x=by,
    color='#BA7A75',
    alpha=.7,
    label='Responses', ax=ax)\
    .set(xlabel='Number of responses', ylabel=None)

ax.tick_params(axis='y', labelsize=8, labelcolor='slategrey')
for bar, label in zip(ax.patches, dr2['invites']):
    y = bar.get_y()
    width = bar.get_width()
    height = bar.get_height()
    ax.text(width+15, y+height, label, ha="center", fontsize=7, color='dimgrey')
ax.axvline(x=df_rates[by].mean(), color='silver', lw=1.5, ls='--')
ax.text(
    110, 35,
    f"Mean: {df_rates[by].mean(): .1f}",
    ha="left", fontsize=9, color='dimgrey')
ax.set_title('Top 10% keywords by responses')
sns.despine()
if 1:
    plt.savefig(f'{plots_path}/responses_keywords_bar.pdf', dpi=300)
plt.show()
