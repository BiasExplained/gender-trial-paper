# %% codecell
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import scipy as sp
import numpy as np
# import researchpy as rp
from matplotlib.ticker import MaxNLocator, MultipleLocator
from collections import Counter
from math import sqrt
import plotly.graph_objects as go

prjwd = "../data/"
savefig = False
plots_path = "plots"


# %% markdown
# # Clean responses and make some checks
# %% codecell
col_names = {'Duration (in seconds)': 'duration_secs',
             '.': 'agreed',
             'Q2': 'credibility',
             'Q6': 'estimated_gender',
             'Q3': 'occupation',
             'Q4': 'participant_gender',
             'Q5': 'feedback'
            }

cat_scores = {'Strongly unlikely': -3,
              'Quite unlikely': -2,
              'Somewhat unlikely': -1,
              'Somewhat likely': 1,
              'Quite likely': 2,
              'Strongly likely': 3,
             }

# %% codecell
fname = "sciofsci_pilot_June-19-2023_00.49.csv"
raw = pd.read_csv(os.path.join(prjwd, fname), header=0)

# %% codecell
df_long = raw
df_long['estimated_gender'] = df_long['estimated_gender'].astype("str")
df_long['egroup'] = df_long['egroup'].astype("category")
df_long['egroup'] = df_long['egroup'].cat.set_categories(['TFemale','Control','TMale'], ordered=True)
df_long['cite_likelyhood_c'] = df_long['cite_likelyhood'].astype("category")

Counter(df_long.cite_likelyhood)
len(set(df_long.ResponseId))

# %% md
# # Final plots
# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]
cmap = 'PuOr'

sns.histplot(data=df_long,
            x="egroup", hue='cite_likelyhood_c', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax = axs[0])\
            .set(title='All responses', xlabel=None, ylabel='Proportion')
sns.histplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", hue='cite_likelyhood_c', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[1])\
            .set(title='Male', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", hue='cite_likelyhood_c', stat='frequency', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[2])\
            .set(title='Female', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", hue='cite_likelyhood_c', stat='frequency', multiple="fill",
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
fig, axs = plt.subplots(1,4, figsize=(5.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]
cmap = 'PuOr'

sns.histplot(data=df_long,
            x="egroup", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax = axs[0])\
            .set(title='All responses', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[1])\
            .set(title='Male', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[2])\
            .set(title='Female', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax= axs[3])\
            .set(title="Uncertain/Don't know", xlabel=None)

for ax in fig.axes:
    ax.tick_params(axis='x', labelrotation=60)

sns.move_legend(axs[3], "upper right", bbox_to_anchor=(1.9, 1.04), frameon=False, title='Cite\n likelihood')
fig.suptitle('Estimated gender')
# fig.supxlabel('Experiment group')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_egroup_proportion_ccs.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]
cl_f, cl_c, cl_m = ["#d6604d", "#d9d9d9", "#00c88a"]
clrs_g = [cl_f, cl_c, cl_m]
sat = 1
sns.boxplot(data=df_long,
            x="egroup", y="cite_likelyhood", width=0.6,
            palette=clrs_g, saturation=sat, ax=axs[0])\
            .set(title='All responses', ylabel='Cite likelihood', xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", y="cite_likelyhood", width=0.6,
            palette=clrs_g, saturation=sat, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", y="cite_likelyhood", width=0.6,
            palette=clrs_g, saturation=sat, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.boxplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", y="cite_likelyhood", width=0.6,
            palette=clrs_g, saturation=sat, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxtplot_egroup_mean-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelyhood",
            palette=clrs_g, errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Mean cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", y="cite_likelyhood",
            palette=clrs_g, errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", y="cite_likelyhood",
            palette=clrs_g, errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", y="cite_likelyhood",
            palette=clrs_g, errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_mean-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(4.7,3.2), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelyhood", estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Median cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

for ax in fig.axes:
    ax.tick_params(axis='x', labelrotation=60)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_median-cite-likelihood_ccs.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,4, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
error = 'sd'

sns.pointplot(data=df_long,
            x="egroup", y="cite_likelyhood", estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Median cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", y="cite_likelyhood",  estimator='median',
            palette=clrs_g, errorbar=error, ax=axs[3])\
            .set(title="Uncertain/Don't know", ylabel=None, xlabel=None)

fig.suptitle('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_egroup_median-cite-likelihood.pdf', dpi=300)
plt.show()

# %%
df_long_ctrl = df_long[(df_long.egroup=='Control') & \
    (df_long.estimated_gender!='Other/Prefer not to say')]\
    .sort_values(by='estimated_gender')
xtlabels = ['Female', 'Male', 'Uncertain']
df_long_ctrl.groupby(['estimated_gender', 'cite_likelyhood_c']).size()
f = (13+10+5)/(3+7+3+13+10+5)
m = (25+5+12)/(17+23+14+25+5+12)
u = (80+50+13)/(73+88+65+80+50+13)

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.histplot(data=df_long_ctrl,
            x="estimated_gender", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax = ax)\
            .set(xlabel='Estimated gender',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.35, 1.04), frameon=False, title='Cite\n likelihood')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion_control.pdf', dpi=300)
plt.show()
# Counter(df_long_ctrl.estimated_gender)
# Counter(df_long.estimated_gender)

# %%
df_long_ctrl[df_long_ctrl.cite_likelyhood > 0].groupby('estimated_gender').count()
df_long_ctrl[df_long_ctrl.sign == 1].groupby('estimated_gender')['cite_likelyhood'].sum()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
# clrs_ge = ["#00c88a", "#f1a340", "#abd9e9"]
cl_u = "#abd9e9"
clrs_ge = [cl_f, cl_m, cl_u]

sns.boxplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelyhood", width=0.6,
            palette=clrs_ge, saturation=sat, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Cite likelihood',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxplot_est-gender_mean-cite-likelihood_control.pdf', dpi=300)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelyhood",
            palette=clrs_ge, errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Mean cite likelihood',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_mean-cite-likelihood_control.pdf', dpi=300)
plt.show()
df_long_ctrl.groupby('estimated_gender').size()
df_long.groupby('estimated_gender').size()
# %%
fig, ax = plt.subplots(figsize=(2.4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="cite_likelyhood", estimator='median',
            palette=clrs_ge, errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Median cite likelihood',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_median-cite-likelihood_control.pdf', dpi=300)
plt.show()

# %%
df_long_estd = df_long[df_long.estimated_gender!='Other/Prefer not to say']\
    .sort_values(by='estimated_gender')

df_long_estd[df_long_estd.egroup=='Control'].groupby(['estimated_gender', 'cite_likelyhood']).size()



# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()
hue_order = [3,2,1,-1,-2,-3]

sns.histplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[0])\
            .set(xlabel=None,
                title='TFemale group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, legend=False, ax=axs[1])\
            .set(xlabel=None,
                title='Control group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", hue='cite_likelyhood_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=cmap, hue_order=hue_order,
            common_norm=False, ax=axs[2])\
            .set(xlabel=None,
                title='TMale group')

for a in axs: a.set_xticklabels(xtlabels)
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
            x="estimated_gender", y="cite_likelyhood", width=0.6,
            palette=clrs_ge, saturation=sat, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Cite likelihood',
                title='TFemale group')
sns.boxplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="cite_likelyhood", width=0.6,
            palette=clrs_ge, saturation=sat, ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.boxplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="cite_likelyhood", width=0.6,
            palette=clrs_ge, saturation=sat, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

for a in axs: a.set_xticklabels(xtlabels)
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/boxtplot_est-gender_proportion.pdf', dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", y="cite_likelyhood",
            palette=clrs_ge, errorbar=error, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Mean cite likelihood',
                title='TFemale group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="cite_likelyhood",
            palette=clrs_ge, errorbar=error,ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="cite_likelyhood",
            palette=clrs_ge, errorbar=error, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

for a in axs: a.set_xticklabels(xtlabels)
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
            x="sign_c", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax = axs[0])\
            .set(title='All responses', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Male"],
            x="sign_c", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax=axs[1])\
            .set(title='Male', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Female"],
            x="sign_c", hue='egroup', stat='count', multiple="dodge",
            discrete=True, shrink=.6, palette=clrs_g, alpha=alpha,
            common_norm=True, legend=False, ax=axs[2])\
            .set(title='Female', xlabel=None)
sns.histplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="sign_c", hue='egroup', stat='count', multiple="dodge",
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
            x="egroup", y="sign",
            palette=clrs_g, errorbar=error, ax=axs[0])\
            .set(title='All responses', ylabel='Mean cite likelihood', xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Male"],
            x="egroup", y="sign",
            palette=clrs_g, errorbar=error, ax=axs[1])\
            .set(title='Male', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Female"],
            x="egroup", y="sign",
            palette=clrs_g, errorbar=error, ax=axs[2])\
            .set(title='Female', ylabel=None, xlabel=None)
sns.pointplot(data=df_long[df_long.estimated_gender == "Uncertain/Don't know"],
            x="egroup", y="sign",
            palette=clrs_g, errorbar=error, ax=axs[3])\
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
            x="estimated_gender", hue='sign_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, alpha=alpha, ax=ax)\
            .set(xlabel='Estimated gender',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.move_legend(ax, "upper right", bbox_to_anchor=(1.4, 1.04), frameon=False, title='Cite\n likelihood')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/histplot_est-gender_proportion-agg_control.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="whitegrid", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, ax = plt.subplots(figsize=(4,3.2), constrained_layout=True)
sns.pointplot(data=df_long_ctrl,
            x="estimated_gender", y="sign",
            palette=clrs_ge, errorbar=error, ax=ax)\
            .set(xlabel='Estimated gender',
                ylabel='Mean cite likelihood',
                title='Control group')
ax.set_xticklabels(xtlabels)
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_mean-cite-likelihood-agg_control.pdf', dpi=300)
plt.show()

# %%
sns.set_theme(style="ticks", rc={'grid.linewidth': 0.5}, font='Arial', font_scale=0.9)
fig, axs = plt.subplots(1,3, figsize=(7.5,3), constrained_layout=True, sharey=True)
axs = axs.flatten()

sns.histplot(data=df_long_estd[df_long_estd.egroup=='TFemale'],
            x="estimated_gender", hue='sign_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, legend=False, alpha=alpha, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Proportion',
                title='TFemale group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", hue='sign_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, legend=False, alpha=alpha, ax=axs[1])\
            .set(xlabel=None,
                title='Control group')
sns.histplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", hue='sign_c', stat='count', multiple="fill",
            discrete=True, shrink=.6, palette=clrs_agg, hue_order=hue_order,
            common_norm=False, alpha=alpha, ax=axs[2])\
            .set(xlabel=None,
                title='TMale group')

for a in axs: a.set_xticklabels(xtlabels)
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
            x="estimated_gender", y="sign",
            palette=clrs_ge, errorbar=error, ax=axs[0])\
            .set(xlabel=None,
                ylabel='Mean cite likelihood',
                title='TFemale group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='Control'],
            x="estimated_gender", y="sign",
            palette=clrs_ge, errorbar=error,ax=axs[1])\
            .set(xlabel=None,
                ylabel=None,
                title='Control group')
sns.pointplot(data=df_long_estd[df_long_estd.egroup=='TMale'],
            x="estimated_gender", y="sign",
            palette=clrs_ge, errorbar=error, ax=axs[2])\
            .set(xlabel=None,
                ylabel=None,
                title='TMale group')

for a in axs: a.set_xticklabels(xtlabels)
fig.supxlabel('Estimated gender')
sns.despine()
if savefig:
    plt.savefig(f'{plots_path}/pointplot_est-gender_proportion-agg.pdf', dpi=300)
plt.show()
