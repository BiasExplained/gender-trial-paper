# %%
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import plot_likert
import textwrap
from collections import Counter
import jsonlines as jsonl
from tqdm import tqdm
# import all data preprocessing steps for (consistency) across analysis
from preprocessing_results import \
df, cat_scores, agg_scores, cred_scores, agg_cred, egroup_l


prjwd = "../data"
savefig = False
plots_path = "plots_likert"


# %%̦
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

# %%
# %% # Answer to R3.8
def label_row(row):
    # FM-No-Recall: TFemale or TMale who did not recall
    if row['egroup'] in ['TFemale', 'TMale'] and row['estimated_gender'] == "Uncertain/Don't know":
        return 'FM-No-Recall'
    # TFemale-Woman: TFemale who guessed female
    elif row['egroup'] == 'TFemale' and row['estimated_gender'] == 'Woman':
        return 'TFemale-Woman'
    # TMale-Man: TMale who guessed male
    elif row['egroup'] == 'TMale' and row['estimated_gender'] == 'Man':
        return 'TMale-Man'
    return None

df['gender_recall_label'] = df.apply(label_row, axis=1)


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
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

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
        if pos_tot == 0:
            pos_tot = [text.get_position()[0] for text in axs.get_xticklabels() if text.get_text() ==
  '0%'][0] * 1.03
        else:
            pos_tot = pos_tot
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
    return axs


# %% Define a dataframe for estimated gender only, H1 and H2 variants
# Answer to R1.2
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]
variable, covariable = 'cite_likelihood', 'estimated_gender'

plot_likert_with_totals(df_est, variable, covariable, cat_scores, cmap, 3.2, 55, 3, -0.25, fs-.5,
    figname=f"likertplot_{variable}_{covariable}_estimated-gender_{pct_txt[pct]}")

# %% Answer to R1.2
variable, covariable = 'credibility', 'estimated_gender'
plot_likert_with_totals(df_est, variable, covariable, cred_scores, cmap4, 3.2, 65, 2, -0.25, fs-.5,
    figname=f"likertplot_{variable}_{covariable}_estimated-gender_{pct_txt[pct]}")



# %% R3.8
savefig = False
variable, covariable = 'cite_likelihood', 'gender_recall_label'
plot_likert_with_totals(df, variable, covariable, cat_scores, cmap, 3, 0, 3, -0.25, fs-.5,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")


# %% Answer to R3.8
variable, covariable = 'credibility', 'gender_recall_label'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 3, 0, 2, -0.25, fs-.5,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")


# %% Plots by keywords - simple grid layout
df_topkw = df[df.keywords.isin(df.groupby('keywords').size().sort_values(ascending=False).head(20).index)]
top_keywords = df_topkw.groupby('keywords').size().sort_values(ascending=False).head(20).index.tolist()



# %%
def plot_likert_grid(df, variable, covariable, grouping_col, group_values, 
                    cat_dic, cmap, nrows, ncols, figsize, figname=''):

      fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, constrained_layout=True)

      for idx, group_value in enumerate(group_values):
          row, col = idx // ncols, idx % ncols
          ax = axes[row, col]

          df_group = df[df[grouping_col] == group_value]

          pivot_table = df_group.pivot_table(
              columns=variable,
              index=covariable,
              values='ResponseId',
              observed=False,
              aggfunc='count')[list(cat_dic.keys())]
          pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

          plot_likert.plot_counts(
              pivot_table,
              cat_dic.keys(),
              bar_labels=True,
              bar_labels_color=bl_color,
              colors=["#ffffff00"] + cmap,
              width=0.6,
              compute_percentages=pct,
              legend=False,
              ax=ax,
              xtick_interval=20
          )

          # adjust percent text
          for text in ax.texts:
              text.set_fontsize(fs)
              text.set_weight(fw)

          # add totals text
          totals = pivot_table.sum(axis=1)
          pos_tot = [text.get_position()[0] for text in ax.get_xticklabels() if text.get_text() == '0%'][0] * 1.03
          for i, (label, total) in enumerate(totals.items()):
              ax.text(pos_tot, len(totals)-1-i+.45, f'N: {total}',
                       va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)

          ax.set_title(textwrap.fill(str(group_value), width=50), fontsize=fs+2)
          if row < nrows-1:
              ax.set_xlabel('')

      handles, labels = ax.get_legend_handles_labels()
      fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -0.05), ncol=len(cat_dic), fontsize=fs+2)

      if savefig:
          fig.savefig(f"{plots_path}/grid_{figname}_{pct_txt[pct]}.pdf",
                     dpi=300, bbox_inches='tight')
      plt.show()


# %%
# Create grid for cite_likelihood
# savefig = True
variable, covariable = 'cite_likelihood', 'keywords'
nrows, ncols = 5, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharey=True, constrained_layout=True)

for idx, keyword in enumerate(top_keywords):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]
    
    df_kw = df_topkw[df_topkw.keywords == keyword]
    
    pivot_table = df_kw.pivot_table(
        columns=variable,
        index='egroup',
        values='ResponseId',
        observed=False,
        aggfunc='count')[list(cat_scores.keys())]
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]
    
    plot_likert.plot_counts(
        pivot_table,
        cat_scores.keys(),
        bar_labels=True,
        bar_labels_color=bl_color,
        colors=["#ffffff00"] + cmap,
        width=0.6,
        compute_percentages=pct,
        legend=False,
        ax=ax,
        xtick_interval=20
    )
    
    # adjust percent text
    for text in ax.texts:
        text.set_fontsize(fs)
        text.set_weight(fw)
    # add totals text
    totals = pivot_table.sum(axis=1)
    pos_tot = [text.get_position()[0] for text in ax.get_xticklabels() if text.get_text() == '0%'][0] * 1.03
    for i, (label, total) in enumerate(totals.items()):
        ax.text(pos_tot, len(totals)-1-i+.45, f'N: {total}',
                 va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)
    
    ax.set_title(textwrap.fill(keyword, width=50), fontsize=fs+2)
    if row < 4:
        ax.set_xlabel('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -0.05), ncol=6, fontsize=fs+2)

if savefig:
    fig.savefig(f"{plots_path}/grid_{variable}_{covariable}_{pct_txt[pct]}.pdf", 
               dpi=300, bbox_inches='tight')
plt.show()


# %%
# Create grid for credibility
variable, covariable = 'credibility', 'keywords'
nrows, ncols = 5, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), sharey=True, constrained_layout=True)

for idx, keyword in enumerate(top_keywords):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]
    
    df_kw = df_topkw[df_topkw.keywords == keyword]
    
    pivot_table = df_kw.pivot_table(
        columns=variable,
        index='egroup',
        values='ResponseId',
        observed=False,
        aggfunc='count')[list(cred_scores.keys())]
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]
    
    plot_likert.plot_counts(
        pivot_table,
        cred_scores.keys(),
        bar_labels=True,
        bar_labels_color=bl_color,
        colors=["#ffffff00"] + cmap4,
        width=0.6,
        compute_percentages=pct,
        legend=False,
        ax=ax,
        xtick_interval=20
    )
    # adjust percent text
    for text in ax.texts:
        text.set_fontsize(fs)
        text.set_weight(fw)
    # add totals text
    totals = pivot_table.sum(axis=1)
    for i, (label, total) in enumerate(totals.items()):
        ax.text(0, len(totals)-1-i+.45, f'N: {total}',
                 va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)
    
    ax.set_title(textwrap.fill(keyword, width=50), fontsize=fs+2)
    if row < 4:
        ax.set_xlabel('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -0.05), ncol=6, fontsize=fs+2)

if savefig:
    fig.savefig(f"{plots_path}/grid_{variable}_{covariable}_{pct_txt[pct]}.pdf", 
               dpi=300, bbox_inches='tight')
plt.show()


# %%
variable, covariable = 'credibility', 'occupation'
plot_likert_with_totals(df, variable, covariable, cred_scores, cmap4, 6, 72, 4, -0.1, fs-2, 17,
    figname=f"likertplot_{variable}_{covariable}_{pct_txt[pct]}")


# -----------------------------------------------------------------------------------------------
# %% get doi by using RecipientEmail and target_author_title_abstract, search in
top_disciplines = df.groupby('discipline').size().sort_values(ascending=False).head(16).index.tolist()
# removed for not having enough responses > History:18, Art:5, Philosophy:2

# %%
# Create grid for cite_likelihood
variable, covariable = 'cite_likelihood', 'discipline'
nrows, ncols = 4, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharey=True, constrained_layout=True)

for idx, keyword in enumerate(top_disciplines):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]
    
    df_kw = df[df.discipline == keyword]
    
    pivot_table = df_kw.pivot_table(
        columns=variable,
        index='egroup',
        values='ResponseId',
        observed=False,
        aggfunc='count')[list(cat_scores.keys())]
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]
    
    plot_likert.plot_counts(
        pivot_table,
        cat_scores.keys(),
        bar_labels=True,
        bar_labels_color=bl_color,
        colors=["#ffffff00"] + cmap,
        width=0.6,
        compute_percentages=pct,
        legend=False,
        ax=ax,
        xtick_interval=20
    )
    
    # adjust percent text
    for text in ax.texts:
        text.set_fontsize(fs)
        text.set_weight(fw)
    # add totals text
    totals = pivot_table.sum(axis=1)
    for i, (label, total) in enumerate(totals.items()):
        ax.text(0, len(totals)-1-i+.45, f'N: {total}',
                 va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)
    
    ax.set_title(textwrap.fill(keyword, width=50), fontsize=fs+2)
    if row < 3:
        ax.set_xlabel('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -0.05), ncol=6, fontsize=fs+2)

if savefig:
    fig.savefig(f"{plots_path}/grid_{variable}_{covariable}_{pct_txt[pct]}.pdf", 
               dpi=300, bbox_inches='tight')
plt.show()


# %%
# Create grid for cite_likelihood
variable, covariable = 'credibility', 'discipline'
nrows, ncols = 4, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharey=True, constrained_layout=True)

for idx, keyword in enumerate(top_disciplines):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]
    
    df_kw = df[df.discipline == keyword]
    
    pivot_table = df_kw.pivot_table(
        columns=variable,
        index='egroup',
        values='ResponseId',
        observed=False,
        aggfunc='count')[list(cred_scores.keys())]
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]
    
    plot_likert.plot_counts(
        pivot_table,
        cred_scores.keys(),
        bar_labels=True,
        bar_labels_color=bl_color,
        colors=["#ffffff00"] + cmap,
        width=0.6,
        compute_percentages=pct,
        legend=False,
        ax=ax,
        xtick_interval=20
    )
    
    # adjust percent text
    for text in ax.texts:
        text.set_fontsize(fs)
        text.set_weight(fw)
    # add totals text
    totals = pivot_table.sum(axis=1)
    for i, (label, total) in enumerate(totals.items()):
        ax.text(0, len(totals)-1-i+.45, f'N: {total}',
                 va='center', ha='left', fontsize=fs, c='#677F92', alpha=.8)
    
    ax.set_title(textwrap.fill(keyword, width=50), fontsize=fs+2)
    if row < 3:
        ax.set_xlabel('')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, -0.05), ncol=6, fontsize=fs+2)

if savefig:
    fig.savefig(f"{plots_path}/grid_{variable}_{covariable}_{pct_txt[pct]}.pdf", 
               dpi=300, bbox_inches='tight')
plt.show()


# %%
