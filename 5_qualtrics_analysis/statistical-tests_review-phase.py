# %%
from IPython.display import display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import Counter
import numpy as np
from scipy.stats import fisher_exact
import scipy.stats
import itertools

# import all data preprocessing steps for (consistency) across analysis
from preprocessing_results import \
df, cat_scores, agg_scores, cred_scores, agg_cred, egroup_l

prjwd = "../data"


# %%
def standardized_effect(u_stat, n1, n2):
    """
    Calculates the Rank-biserial correlation from the U-test statistic
    (https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Rank-biserial_correlation)
    """
    return 1-((2*u_stat)/(n1*n2))


def run_mannwhitneyu(c_values, f_values, m_values):
    """
    This is the U test to compare distributions. Use the alternative=less and input the first and second
    parameters accordingly, to measure the hypothesis of C<M and F<M.
    inputs:
        c_values: list of values for control group
        f_values: list of values for female group
        m_values: list of values for male group 
    """
    c_size = len(c_values)
    f_size = len(f_values)
    m_size = len(m_values)
    # validation that the test works as expected
    # U_test_cc = scipy.stats.mannwhitneyu(c_values, c_values, alternative='less')
    # U_test_cc_stat, U_test_cc_p = standardized_effect(U_test_cc.statistic, c_size, c_size), U_test_cc.pvalue
    # print(f'C<C (validation): {U_test_cc_stat:.4f} (p={U_test_cc_p:.4f})')
    
    # The U-test gives only a p-value, this other measurement gives the magnitude of the effect
    U_test_cm = scipy.stats.mannwhitneyu(c_values, m_values, alternative='less')
    U_test_cm_stat, U_test_cm_p = standardized_effect(U_test_cm.statistic, c_size, m_size), U_test_cm.pvalue
    U_test_cf = scipy.stats.mannwhitneyu(f_values, c_values, alternative='less')
    U_test_cf_stat, U_test_cf_p = standardized_effect(U_test_cf.statistic, c_size, f_size), U_test_cf.pvalue
    U_test_fm = scipy.stats.mannwhitneyu(f_values, m_values, alternative='less')
    U_test_fm_stat, U_test_fm_p = standardized_effect(U_test_fm.statistic, f_size, m_size), U_test_fm.pvalue

    print(f'C<M: {U_test_cm_stat:.4f} (p={U_test_cm_p:.4f})')
    print(f'F<C: {U_test_cf_stat:.4f} (p={U_test_cf_p:.4f})')
    print(f'F<M: {U_test_fm_stat:.4f} (p={U_test_fm_p:.4f})')


def run_fisher_exact(df, variable, covariable, dic_scores, scale_point, bonferroni_correction_factor=1):
    """
    # this is the bonferroni correction, which should be equal to the number of contingency tables we have below
    # e.g., if we compare F<M, F<C, and C<M, we need a bonferroni of 3
    
    """
    pivot_table = df.pivot_table(
            columns=variable,
            index=covariable,
            values='ResponseId',
            observed=True,
            aggfunc='count')[list(dic_scores.keys())]
    pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

    # count how many respondents have selected the top scores on the likert scale and how many did not
    c_topscore = pivot_table[scale_point]['Control']
    f_topscore = pivot_table[scale_point]['TFemale']
    m_topscore = pivot_table[scale_point]['TMale']
    c_not_topscore = pivot_table.sum(axis=1)['Control'] - c_topscore
    f_not_topscore = pivot_table.sum(axis=1)['TFemale'] - f_topscore
    m_not_topscore = pivot_table.sum(axis=1)['TMale'] - m_topscore

    # build a contingency table out of the numbers above
    table_fc = np.array([[f_topscore, f_not_topscore], [c_topscore, c_not_topscore]])
    table_cm = np.array([[c_topscore, c_not_topscore], [m_topscore, m_not_topscore]])
    table_fm = np.array([[f_topscore, f_not_topscore], [m_topscore, m_not_topscore]])

    res_fc = fisher_exact(table_fc, alternative='less')
    res_cm = fisher_exact(table_cm, alternative='less')
    res_fm = fisher_exact(table_fm, alternative='less')

    print(f'C<M: {res_cm.statistic:.4f}, (p={res_cm.pvalue*bonferroni_correction_factor:.4f})')
    print(f'F<C: {res_fc.statistic:.4f}, (p={res_fc.pvalue*bonferroni_correction_factor:.4f})')
    print(f'F<M: {res_fm.statistic:.4f}, (p={res_fm.pvalue*bonferroni_correction_factor:.4f})')


# ------------------------------------------------------------------------------------------
# %% Additional robustness check:
# Group respondents by the author gender they perceive, not the gender condition that
# they are assigned to. C=Other/Prefer not to say, F=Woman, M=Man
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]

# Man Whitney U test for Cite Likelihood and credibility
variable, covariable = 'cite_likelihood_num', 'estimated_gender'
c_values = df_est[df_est[covariable]=='Other/Prefer not to say'][variable].tolist()
f_values = df_est[df_est[covariable]=='Woman'][variable].tolist()
m_values = df_est[df_est[covariable]=='Man'][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

variable, covariable = 'credibility_num', 'estimated_gender'
c_values = df_est[df_est[covariable]=='Other/Prefer not to say'][variable].tolist()
f_values = df_est[df_est[covariable]=='Woman'][variable].tolist()
m_values = df_est[df_est[covariable]=='Man'][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

# %% 
variable, covariable = 'cite_likelihood', 'estimated_gender'
dic_scores = cat_scores
scale_point = 'Strongly likely'
bonferroni_correction_factor = 3

pivot_table = df_est.pivot_table(
            columns=variable,
            index=covariable,
            values='ResponseId',
            observed=True,
            aggfunc='count')[list(dic_scores.keys())]
pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

c_topscore = pivot_table[scale_point]['Other/Prefer not to say']
f_topscore = pivot_table[scale_point]['Woman']
m_topscore = pivot_table[scale_point]['Man']
c_not_topscore = pivot_table.sum(axis=1)['Other/Prefer not to say'] - c_topscore
f_not_topscore = pivot_table.sum(axis=1)['Woman'] - f_topscore
m_not_topscore = pivot_table.sum(axis=1)['Man'] - m_topscore

# build a contingency table out of the numbers above
table_fc = np.array([[f_topscore, f_not_topscore], [c_topscore, c_not_topscore]])
table_cm = np.array([[c_topscore, c_not_topscore], [m_topscore, m_not_topscore]])
table_fm = np.array([[f_topscore, f_not_topscore], [m_topscore, m_not_topscore]])

res_fc = fisher_exact(table_fc, alternative='less')
res_cm = fisher_exact(table_cm, alternative='less')
res_fm = fisher_exact(table_fm, alternative='less')

print(f'C<M: {res_cm.statistic:.4f}, (p={res_cm.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<C: {res_fc.statistic:.4f}, (p={res_fc.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<M: {res_fm.statistic:.4f}, (p={res_fm.pvalue*bonferroni_correction_factor:.4f})')

# %%
variable, covariable = 'credibility', 'estimated_gender'
dic_scores = cred_scores
scale_point = 'Very credible'
bonferroni_correction_factor = 3

pivot_table = df_est.pivot_table(
            columns=variable,
            index=covariable,
            values='ResponseId',
            observed=True,
            aggfunc='count')[list(dic_scores.keys())]
pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

c_topscore = pivot_table[scale_point]['Other/Prefer not to say']
f_topscore = pivot_table[scale_point]['Woman']
m_topscore = pivot_table[scale_point]['Man']
c_not_topscore = pivot_table.sum(axis=1)['Other/Prefer not to say'] - c_topscore
f_not_topscore = pivot_table.sum(axis=1)['Woman'] - f_topscore
m_not_topscore = pivot_table.sum(axis=1)['Man'] - m_topscore

# build a contingency table out of the numbers above
table_fc = np.array([[f_topscore, f_not_topscore], [c_topscore, c_not_topscore]])
table_cm = np.array([[c_topscore, c_not_topscore], [m_topscore, m_not_topscore]])
table_fm = np.array([[f_topscore, f_not_topscore], [m_topscore, m_not_topscore]])

res_fc = fisher_exact(table_fc, alternative='less')
res_cm = fisher_exact(table_cm, alternative='less')
res_fm = fisher_exact(table_fm, alternative='less')

print(f'C<M: {res_cm.statistic:.4f}, (p={res_cm.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<C: {res_fc.statistic:.4f}, (p={res_fc.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<M: {res_fm.statistic:.4f}, (p={res_fm.pvalue*bonferroni_correction_factor:.4f})')


# ------------------------------------------------------------------------------------------
# %% Responses stratified by discipline
dic_scores = cat_scores
variable, covariable = 'cite_likelihood_num', 'egroup'
top_disciplines = df.groupby('discipline').size().sort_values(ascending=False).head(16).index.tolist()

for discipline in top_disciplines:
    print(f'\nDiscipline: {discipline} - {variable}')
    df_discipline = df[df['discipline'] == discipline]

    # Test Cite Likelihood
    c_values = df_discipline[df_discipline[covariable]=='Control'][variable].tolist()
    f_values = df_discipline[df_discipline[covariable]=='TFemale'][variable].tolist()
    m_values = df_discipline[df_discipline[covariable]=='TMale'][variable].tolist()
    run_mannwhitneyu(c_values, f_values, m_values)

    # Fisher test for citation likelihood
    run_fisher_exact(df_discipline, 'cite_likelihood', covariable, dic_scores, 'Strongly likely', bonferroni_correction_factor=3)

# %% Test Credibility
dic_scores = cred_scores
variable, covariable = 'credibility_num', 'egroup'

for discipline in top_disciplines:
    print(f'\nDiscipline: {discipline} - {variable}')
    # remove null values
    df_discipline = df[(df['discipline'] == discipline) & (df[variable].notnull())]

    # Test Cite Likelihood
    c_values = df_discipline[df_discipline[covariable]=='Control'][variable].tolist()
    f_values = df_discipline[df_discipline[covariable]=='TFemale'][variable].tolist()
    m_values = df_discipline[df_discipline[covariable]=='TMale'][variable].tolist()
    run_mannwhitneyu(c_values, f_values, m_values)

    # Fisher test for citation likelihood
    run_fisher_exact(df_discipline, 'credibility', covariable, dic_scores, 'Very credible', bonferroni_correction_factor=3)


# ------------------------------------------------------------------------------------------
# %% Responses stratified by keyword pairs (abstract group)
dic_scores = cat_scores
variable, covariable = 'cite_likelihood_num', 'egroup'
top_keywords = df.groupby('keywords').size().sort_values(ascending=False).head(16).index.tolist()

for keywords in top_keywords:
    print(f'\nDiscipline: {keywords} - {variable}')
    df_keywords = df[df['keywords'] == keywords]

    # Test Cite Likelihood
    c_values = df_keywords[df_keywords[covariable]=='Control'][variable].tolist()
    f_values = df_keywords[df_keywords[covariable]=='TFemale'][variable].tolist()
    m_values = df_keywords[df_keywords[covariable]=='TMale'][variable].tolist()
    try:
        run_mannwhitneyu(c_values, f_values, m_values)
    except:
        print(f"Mann-Whitney U test for {keywords} failed")

    # Fisher test for citation likelihood
    try:
        run_fisher_exact(df_keywords, 'cite_likelihood', covariable, dic_scores, 'Strongly likely', bonferroni_correction_factor=3)
    except:
        print(f"Fisher exact test for {keywords} failed")

# %% Test Credibility
dic_scores = cred_scores
variable, covariable = 'credibility_num', 'egroup'

for keywords in top_keywords:
    print(f'\nDiscipline: {keywords} - {variable}')
    # remove null values
    df_keywords = df[(df['keywords'] == keywords) & (df[variable].notnull())]

    # Test Cite Likelihood
    c_values = df_keywords[df_keywords[covariable]=='Control'][variable].tolist()
    f_values = df_keywords[df_keywords[covariable]=='TFemale'][variable].tolist()
    m_values = df_keywords[df_keywords[covariable]=='TMale'][variable].tolist()
    try:
        run_mannwhitneyu(c_values, f_values, m_values)
    except:
        print(f"Mann-Whitney U test for {keywords} failed")

    # Fisher test for citation likelihood
    try:
        run_fisher_exact(df_keywords, 'credibility', covariable, dic_scores, 'Very credible', bonferroni_correction_factor=3)
    except:
        print(f"Fisher exact test for {keywords} failed")


# ------------------------------------------------------------------------------------------
# %% Effect of recalling the author gender. Ratings from
# participants who encoded gender information to participants who could not remember the
# author’s gender.

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

# %% Test Cite Likelihood
variable, covariable = 'cite_likelihood_num', 'gender_recall_label'
c_values = df[df[covariable]=='FM-No-Recall'][variable].tolist()
f_values = df[df[covariable]=='TFemale-Woman'][variable].tolist()
m_values = df[df[covariable]=='TMale-Man'][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

# %%Test Credibility
variable, covariable = 'credibility_num', 'gender_recall_label'
c_values = df[(df[covariable]=='FM-No-Recall') & (df[variable].notnull())][variable].tolist()
f_values = df[(df[covariable]=='TFemale-Woman') & (df[variable].notnull())][variable].tolist()
m_values = df[(df[covariable]=='TMale-Man') & (df[variable].notnull())][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

# %% Fisher test for citation likelihood
variable, covariable = 'cite_likelihood', 'gender_recall_label'
dic_scores = cat_scores
scale_point = 'Strongly likely'
bonferroni_correction_factor = 3

pivot_table = df.pivot_table(
            columns=variable,
            index=covariable,
            values='ResponseId',
            observed=True,
            aggfunc='count')[list(dic_scores.keys())]
pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

c_topscore = pivot_table[scale_point]['FM-No-Recall']
f_topscore = pivot_table[scale_point]['TFemale-Woman']
m_topscore = pivot_table[scale_point]['TMale-Man']
c_not_topscore = pivot_table.sum(axis=1)['FM-No-Recall'] - c_topscore
f_not_topscore = pivot_table.sum(axis=1)['TFemale-Woman'] - f_topscore
m_not_topscore = pivot_table.sum(axis=1)['TMale-Man'] - m_topscore

# build a contingency table out of the numbers above
table_fc = np.array([[f_topscore, f_not_topscore], [c_topscore, c_not_topscore]])
table_cm = np.array([[c_topscore, c_not_topscore], [m_topscore, m_not_topscore]])
table_fm = np.array([[f_topscore, f_not_topscore], [m_topscore, m_not_topscore]])

res_fc = fisher_exact(table_fc, alternative='less')
res_cm = fisher_exact(table_cm, alternative='less')
res_fm = fisher_exact(table_fm, alternative='less')

print(f'C<M: {res_cm.statistic:.4f}, (p={res_cm.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<C: {res_fc.statistic:.4f}, (p={res_fc.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<M: {res_fm.statistic:.4f}, (p={res_fm.pvalue*bonferroni_correction_factor:.4f})')

# %%
variable, covariable = 'credibility', 'gender_recall_label'
dic_scores = cred_scores
scale_point = 'Very credible'
bonferroni_correction_factor = 3

pivot_table = df.pivot_table(
            columns=variable,
            index=covariable,
            values='ResponseId',
            observed=True,
            aggfunc='count')[list(dic_scores.keys())]
pivot_table = pivot_table.loc[pivot_table.sum(axis=1).sort_values(ascending=False).index]

c_topscore = pivot_table[scale_point]['FM-No-Recall']
f_topscore = pivot_table[scale_point]['TFemale-Woman']
m_topscore = pivot_table[scale_point]['TMale-Man']
c_not_topscore = pivot_table.sum(axis=1)['FM-No-Recall'] - c_topscore
f_not_topscore = pivot_table.sum(axis=1)['TFemale-Woman'] - f_topscore
m_not_topscore = pivot_table.sum(axis=1)['TMale-Man'] - m_topscore

# build a contingency table out of the numbers above
table_fc = np.array([[f_topscore, f_not_topscore], [c_topscore, c_not_topscore]])
table_cm = np.array([[c_topscore, c_not_topscore], [m_topscore, m_not_topscore]])
table_fm = np.array([[f_topscore, f_not_topscore], [m_topscore, m_not_topscore]])

res_fc = fisher_exact(table_fc, alternative='less')
res_cm = fisher_exact(table_cm, alternative='less')
res_fm = fisher_exact(table_fm, alternative='less')

print(f'C<M: {res_cm.statistic:.4f}, (p={res_cm.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<C: {res_fc.statistic:.4f}, (p={res_fc.pvalue*bonferroni_correction_factor:.4f})')
print(f'F<M: {res_fm.statistic:.4f}, (p={res_fm.pvalue*bonferroni_correction_factor:.4f})')


# ------------------------------------------------------------------------------------------
# %% General statistical tests for main results
# Test Cite Likelihood
variable, covariable = 'cite_likelihood_num', 'estimated_gender'
# Filter Control group only, exclude Uncertain
df_control = df[df['egroup'] == 'Control']

control_dic = df_control.pivot_table(
    columns=variable,
    index=covariable,
    values='ResponseId',
    observed=True,
    aggfunc='count').sum(axis=1).to_dict()

man_responses = control_dic.get('Man')
woman_responses = control_dic.get('Woman')
control_total = sum(control_dic.values())

contingency = np.array([
    [woman_responses, man_responses],
    [control_total - woman_responses, control_total - man_responses]
    ])
# contingency = np.array([
#     [woman_responses, control_total - woman_responses],
#     [man_responses, control_total - man_responses]
#     ])

odds_ratio, p_value = fisher_exact(contingency, alternative='less')

print(f'OR: {odds_ratio:.4f}, p-value: {p_value:.4f}')

# ------------------------------------------------------------------------------------------
# %% Stat tests for Overal results, H1 nd H2
# Man Whitney U test for Cite Likelihood and credibility
variable, covariable = 'cite_likelihood_num', 'egroup'
c_values = df[df[covariable]=='Control'][variable].tolist()
f_values = df[df[covariable]=='TFemale'][variable].tolist()
m_values = df[df[covariable]=='TMale'][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

variable, covariable = 'credibility_num', 'egroup'
c_values = df[(df[covariable]=='Control') & (df[variable].notnull())][variable].tolist()
f_values = df[(df[covariable]=='TFemale') & (df[variable].notnull())][variable].tolist()
m_values = df[(df[covariable]=='TMale') & (df[variable].notnull())][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

# Fisher test for citation likelihood
run_fisher_exact(df, 'cite_likelihood', covariable, cat_scores, 'Strongly likely', bonferroni_correction_factor=3)
# Fisher test for credibility
run_fisher_exact(df, 'credibility', covariable, cred_scores, 'Very credible', bonferroni_correction_factor=3)


# %%
# Repeat for estimated gender only, gamma subsets
df_est = df[df['estimated_gender'].ne("Uncertain/Don't know")]

variable, covariable = 'cite_likelihood_num', 'egroup'
c_values = df_est[df_est[covariable]=='Control'][variable].tolist()
f_values = df_est[df_est[covariable]=='TFemale'][variable].tolist()
m_values = df_est[df_est[covariable]=='TMale'][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

variable, covariable = 'credibility_num', 'egroup'
c_values = df_est[(df_est[covariable]=='Control') & (df_est[variable].notnull())][variable].tolist()
f_values = df_est[(df_est[covariable]=='TFemale') & (df_est[variable].notnull())][variable].tolist()
m_values = df_est[(df_est[covariable]=='TMale') & (df_est[variable].notnull())][variable].tolist()
run_mannwhitneyu(c_values, f_values, m_values)

# Fisher test for citation likelihood
run_fisher_exact(df_est, 'cite_likelihood', covariable, cat_scores, 'Strongly likely', bonferroni_correction_factor=3)
# Fisher test for credibility
run_fisher_exact(df_est, 'credibility', covariable, cred_scores, 'Very credible', bonferroni_correction_factor=3)

# %%
