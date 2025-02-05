from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import plot_likert
import textwrap
from collections import Counter
# import all data preprocessing steps for (consistency) across analysis
from preprocessing_results import \
df, cat_scores, agg_scores, cred_scores, agg_cred, egroup_l


prjwd = "../data"
savefig = False
plots_path = "plots_likert"


# %%
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
                      break_long_words=break_long_words))
    if axis == 'y':
        ax.set_yticklabels(labels, rotation=0)
    else:
        ax.set_xticklabels(labels, rotation=0)

def cap(txt):
    if txt.startswith('eg'):
        return "Groups"
    else:
        return txt.replace('_', ' ').replace('-',' ').capitalize()

# %%
display(Counter(df.credibility))
display(Counter(df.cite_likelihood))
display(Counter(df[df['egroup'] == 'TMale']['cite_likelihood']))

# %%
display(df.pivot_table(
    columns='cite_likelihood',
    index='egroup',
    values='ResponseId',
    aggfunc='count',
    dropna=True,
    observed=True)
    )

display(df.pivot_table(
    columns='credibility',
    index='egroup',
    values='ResponseId',
    aggfunc='count',
    observed=True
    ).sort_index(axis=1, level=lambda x: cred_scores.get(x))
    )

# %% --------------------------------------------------------------------------
# define ploting params
sns.set_theme(
    style="ticks",
    rc={'patch.edgecolor': 'none'},
    font='Arial',
    font_scale=0.84)
cl_f, cl_c, cl_m = ["#d6604d", "#d9d9d9", "#00c88a"]
hue_order = [3, 2, 1, -1, -2, -3]
cmap = sns.color_palette('PuOr', 6)
cmap2 = sns.color_palette('PuOr', 2)
cmap4 = sns.color_palette('PuOr', 4)
bl_color = "k"
fs = 8
fw = 'light'
pct = True
pct_txt = {True: "percent", False: "count"}
pct_fmt = {True: "%.1f", False: "%d"}
plot_likert.__internal__.BAR_LABEL_FORMAT = pct_fmt[pct]

# %% md
# # Plots for Cite likelihood

# %%
def plot_likert_with_totals(df, variable, covariable, cat_dic, cmap, fig_h,
    pos_tot, ncols, bbox_y, fsl, wrap_len=0, wrap_ax='y', figname='',
    figtitle=''):

    pivot_table = df.pivot_table(
        columns=variable,
        index=covariable,
        values='ResponseId',
        observed=True,
        aggfunc='count')[list(cat_dic.keys())]

    fig, axs = plt.subplots(figsize=(5, fig_h), constrained_layout=True)
    plot_likert.plot_counts(
        pivot_table,
        cat_dic.keys(),
        bar_labels=True,
        bar_labels_color=bl_color,
        colors=["#ffffff00"] + cmap,
        width=0.6,
        compute_percentages=pct,
        legend=False,
        ax=axs
        )
    # adjust percent text
    for text in axs.texts:
        text.set_fontsize(fs)
        text.set_weight(fw)
    # add totals text
    totals = pivot_table.sum(axis=1)
    for i, (label, total) in enumerate(totals.items()):
        axs.text(pos_tot, len(totals)-1-i+.45, f'N: {total}',
                 va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)
    # break lines aof long y-axis labels
    if wrap_len > 0:
        wrap_labels(axs, wrap_len, wrap_ax)

    axs.set_title(cap(variable) + f"{figtitle}")
    axs.set_ylabel(cap(covariable))

    axs.legend(loc='upper center', bbox_to_anchor=(0.5, bbox_y),
        ncol=ncols, fontsize=fsl)

    sns.despine()

    if savefig:
        fig.savefig(f"{plots_path}/{figname}.pdf", dpi=300, bbox_inches = 'tight')
    plt.show()


# %%
variable, covariable = 'cite_likelihood', 'egroup'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 2.5, 57, 3, -0.37, fs,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")
# %%
variable, covariable = 'cite_likelihood', 'estimated_gender'
plot_likert_with_totals(df[df.egroup.eq('Control')], variable, covariable, cat_scores, cmap, 3.2, 60, 3, -0.25, fs,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}_control")

# %%
variable, covariable = ('sign', 'egroup')
plot_likert_with_totals(df, variable, covariable, agg_scores, cmap2, 2.5, 57, 2, -0.3, fs,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %% Define a dataframe for estimated gender only, H1 and H2 variants
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]
variable, covariable = 'cite_likelihood', 'egroup'

plot_likert_with_totals(df_est, variable, covariable, cat_scores, cmap, 2.5, 52, 3, -0.4, fs,
    figname=f"likertplot_{variable}_{covariable}_estimated-gender_{pct_txt[pct]}")

# %%
variable, covariable = 'cite_likelihood', 'estimated_gender'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 3.2, 60, 3, -0.25, fs-1.5,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")


# %%
variable, covariable = 'cite_likelihood', 'occupation'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 6, 72, 3, -0.1, fs-1, 20,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
variable, covariable = 'cite_likelihood', 'participant_gender'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 3.2, 85, 3, -0.25, fs-.5,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
variable, covariable = 'cite_likelihood', 'estimated_gender_reason'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 4.5, 70, 3, -0.16, fs-1, 20,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %% md
# # Plots for credibility

# %%
variable, covariable = 'credibility', 'egroup'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 2.5, 67, 4, -0.37, fs-1,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")


# %%
variable, covariable = 'credibility_sign', 'egroup'
plot_likert_with_totals(df, variable, covariable, agg_cred, cmap2, 2.5, 66, 2, -0.37, fs-1,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
variable, covariable = 'credibility', 'egroup'
plot_likert_with_totals(df_est, variable, covariable, cred_scores, cmap4, 2.5, 62, 4, -0.37, fs-1,
    figname=f"likertplot_{variable}_{covariable}_estimated-gender_{pct_txt[pct]}")

# %%
Counter(df.credibility)
df.pivot_table(
    columns=variable,
    index=covariable,
    values='ResponseId',
    observed=True,
    aggfunc='count')[list(cred_scores.keys())]

# %%
variable, covariable = 'credibility', 'estimated_gender'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 3.2, 67, 4, -0.25, fs-1.5, 15,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")
# %%
variable, covariable = 'credibility', 'estimated_gender_reason'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 4.5, 70, 3, -0.16, fs-1, 20,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
variable, covariable = 'credibility', 'occupation'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 6, 72, 4, -0.1, fs-2, 17,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
variable, covariable = 'credibility', 'participant_gender'
savefig=True
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 2.9, 75, 4, -0.25, fs-1.7, 12,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")

# %%
# size() counts all elements, including NaN values.
# count() counts only non-null (valid) values, excluding NaN values.

# %% md
# # Plots for All groups by estimated gender
# %%
fig, axs = plt.subplots(figsize=(7, 6), constrained_layout=True, sharey=True)
variable = ['egroup', 'estimated_gender']
covariable = 'cite_likelihood'
pivot_table = pd.pivot_table(
    df,
    index=variable,
    columns=covariable,
    values='ResponseId',
    observed=True,
    aggfunc='count')[list(cat_scores.keys())]

plot_likert.plot_counts(
    pivot_table,
    cat_scores.keys(),
    bar_labels=True,
    bar_labels_color=bl_color,
    colors=["#ffffff00"] + cmap,
    width=0.7,
    compute_percentages=pct,
    ax=axs
    )
for text in axs.texts:
    text.set_fontsize(fs), text.set_weight(fw)

wrap_labels(axs, 20, 'y')
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)

totals = pivot_table.sum(axis=1)
for i, (label, total) in enumerate(totals.items()):
    axs.text(65, len(totals)-1-i+.45, f'N: {total}',
             va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)

sns.despine()
axs.set_ylabel(cap(variable[0]) +', '+ cap(variable[1]))
axs.set_title(cap(covariable))
if savefig:
    plt.savefig(
        f'{plots_path}/likertplot_{"-".join(variable)}_{covariable}_{pct_txt[pct]}.pdf',
        dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(7, 6.5), constrained_layout=True, sharey=True)
variable = ['egroup', 'estimated_gender']
covariable = 'credibility'
pivot_table = pd.pivot_table(
    df,
    index=variable,
    columns=covariable,
    values='ResponseId',
    observed=True,
    aggfunc='count')[list(cred_scores.keys())]

plot_likert.plot_counts(
    pivot_table,
    cred_scores.keys(),
    bar_labels=True,
    bar_labels_color=bl_color,
    colors=["#ffffff00"] + cmap4,
    width=0.7,
    compute_percentages=pct,
    ax=axs
    )
for text in axs.texts:
    text.set_fontsize(fs), text.set_weight(fw)

wrap_labels(axs, 20, 'y')
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)

sns.despine()
axs.set_ylabel(cap(variable[0]) +', '+ cap(variable[1]))
axs.set_title(cap(covariable))

totals = pivot_table.sum(axis=1)
for i, (label, total) in enumerate(totals.items()):
    axs.text(72, len(totals)-1-i+.45, f'N: {total}',
             va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)

if savefig:
    plt.savefig(
        f'{plots_path}/likertplot_{"-".join(variable)}_{covariable}_{pct_txt[pct]}.pdf',
        dpi=300)
plt.show()

# %% md
# # Plots for largest keyword group
# %%
kw1 = 'Climate-change_Affordable-housing'
df_kw1 = df[df.keywords.eq(kw1)]
def title(txt): return txt.replace('-',' ').replace('_',', ')

# %%
variable, covariable = 'cite_likelihood', 'egroup'
plot_likert_with_totals(df_kw1, variable, covariable, cat_scores, cmap, 2.5, 69, 3, -0.37, fs,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}_{kw1}", figtitle=f" - {title(kw1)}" )
# %%
variable, covariable = 'credibility', 'egroup'
plot_likert_with_totals(df_kw1, variable, covariable, cred_scores, cmap4, 2.5, 62, 4, -0.37, fs-1,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}_{kw1}", figtitle=f" - {title(kw1)}" )


# %% md
# # Plots Heatmaps for credbility and cite likelihood

# %%
display(pd.pivot_table(
    df,
    index=['egroup', 'estimated_gender'],
    columns='cite_likelihood',
    values='cite_likelihood_num',
    observed=True,
    aggfunc='count')
    )

display(pd.pivot_table(
    df,
    index='cite_likelihood',
    columns='egroup',
    values='cite_likelihood_num',
    observed=True,
    aggfunc='count')
    )
# %%
# df_wide = pd.pivot_table(df_long, index='cite_likelihood', columns='egroup', values='cite_likelihood_num', aggfunc='count')
fig, axs = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
variable = 'cite_likelihood'
covariable = 'egroup'

sns.heatmap(
    pd.pivot_table(
        df,
        index=variable,
        columns=covariable,
        values='cite_likelihood_num',
        observed=True,
        aggfunc='count').iloc[[4, 0, 2, 3, 1, 5]],
    annot=True,
    fmt="3d",
    cmap=sns.cubehelix_palette(as_cmap=True))

axs.set_ylabel(cap(variable))
axs.set_xlabel(cap(covariable))
if savefig:
    plt.savefig(
        f'{plots_path}/heatmap_{variable}_{covariable}_{pct_txt[pct]}.pdf',
        dpi=300)
plt.show()

# %%
fig, axs = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
variable = 'credibility'
covariable = 'egroup'

sns.heatmap(
    pd.pivot_table(
        df,
        index=variable,
        columns=covariable,
        values='cite_likelihood_num',
        observed=True,
        aggfunc='count').iloc[[3, 1, 2, 0]],
    annot=True,
    fmt="3d",
    cmap=sns.cubehelix_palette(as_cmap=True))

axs.set_ylabel(cap(variable))
axs.set_xlabel(cap(covariable))
if savefig:
    plt.savefig(
        f'{plots_path}/heatmap_{variable}_{covariable}_{pct_txt[pct]}.pdf',
        dpi=300)
plt.show()
