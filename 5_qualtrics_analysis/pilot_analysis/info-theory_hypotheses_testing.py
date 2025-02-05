# %% codecell
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import scipy as sp
import numpy as np
import statsmodels.api as sm
import researchpy as rp
from matplotlib.ticker import MaxNLocator, MultipleLocator
from collections import Counter
from math import sqrt


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

egroup_num = {'Control': 1,
              'TFemale': 2,
              'TMale': 3
              }

# %% Pre-processing and cleaning steps of original data
# Read the CSV file into a DataFrame
fname = "sciofsci_pilot_June-19-2023_00.49.csv"
raw = pd.read_csv(os.path.join(prjwd, fname), header=0)
original = False

if original:
    # Drop extra headers
    raw.drop([0, 1], inplace=True)

    # Rename columns using the provided dictionary
    raw = raw.rename(columns=col_names)

    # Filter out responses where participants did not agree to participate
    raw = raw[raw.agreed == 'Yes, I want to participate in this survey']

    # Remove unfinished responses
    raw = raw[raw.Finished == 'True']

    # Select specific columns from the DataFrame
    raw[['ResponseId','estimated_gender','TFemale','Control','TMale']].head(3)

    # Select specific columns from the raw DataFrame and create a copy
    df = raw[[
        'ResponseId','duration_secs','credibility',
        'estimated_gender','occupation','participant_gender',
        'TFemale','Control','TMale'
        ]].copy()

    # Convert categorical scores to numeric values
    df['Control'] = df['Control'].apply(lambda x: cat_scores[x] if isinstance(x, str) else x)
    df['TFemale'] = df['TFemale'].apply(lambda x: cat_scores[x] if isinstance(x, str) else x)
    df['TMale'] = df['TMale'].apply(lambda x: cat_scores[x] if isinstance(x, str) else x)

    # Reshape the DataFrame from wide to long format
    df_long = df[['ResponseId','estimated_gender','TFemale','Control','TMale']] \
            .melt(id_vars=['ResponseId','estimated_gender'],
                var_name='egroup',
                value_name='cite_likelyhood') \
            .reset_index(drop=True)

    # Create bins and labels for categorizing the 'cite_likelyhood' values
    bins = [-4, 0, 3]
    names = ['Unlikely', 'Likely']

    # Categorize the 'cite_likelyhood' values based on the bins and labels
    df_long['sign_c'] = pd.cut(df_long['cite_likelyhood'], bins, labels=names)

    # Assign a sign (1 or -1) based on the 'cite_likelyhood' values
    df_long['sign'] = df_long['cite_likelyhood'].apply(lambda x: 1 if x > 0 else -1)

    # Map the 'egroup' values to numeric values
    df_long['egroup_n'] = df_long['egroup'].apply(lambda x: egroup_num[x])


# %% markdown
# ### Quick validations (optional)
# %% codecell
print(df.Control.sum(), df.TFemale.sum(), df.TMale.sum())
print(df.Control.sum() + df.TFemale.sum() + df.TMale.sum())
print(df_long.cite_likelyhood.sum())

# %% codecell
Counter(raw.TFemale), Counter(raw.TMale), Counter(raw.Control)
c = Counter(raw['estimated_gender'])
v = list(c.values())
display(
    c,
    v[0]/sum(v),
    [f'{i/sum(v):.1%}' for i in v],
    sum(Counter(raw['estimated_gender']).values()),
    v)

# %% codecell
# Drop rows with missing values and create a copy of the DataFrame
df_long = raw

# Convert 'estimated_gender' column to string data type
df_long['estimated_gender'] = df_long['estimated_gender'].astype("str")

# Convert 'egroup' column to categorical data type and set the categories in a specific order
df_long['egroup'] = df_long['egroup'].astype("category")
df_long['egroup'] = df_long['egroup'].cat.set_categories(['TFemale','Control','TMale'], ordered=True)

# Convert 'cite_likelyhood' column to categorical data type
df_long['cite_likelyhood_c'] = df_long['cite_likelyhood'].astype("category")

# Display the count of each unique value in 'cite_likelyhood' column and the number of unique 'ResponseId'
display(
    Counter(df_long.cite_likelyhood),
    len(set(df_long.ResponseId)))

# %%
display(df_long[df_long.egroup=='TFemale']['cite_likelyhood'].mean(), \
    df_long[df_long.egroup=='TFemale']['cite_likelyhood'].std())
display(df_long[df_long.egroup=='TMale']['cite_likelyhood'].mean(), \
    df_long[df_long.egroup=='TMale']['cite_likelyhood'].std())


# %%
df_long_ctrl = df_long[(df_long.egroup=='Control') & \
    (df_long.estimated_gender!='Other/Prefer not to say')]\
    .sort_values(by='estimated_gender')

display(df_long_ctrl[df_long_ctrl.cite_likelyhood > 0].groupby('estimated_gender').count())
display(df_long_ctrl[df_long_ctrl.sign == 1].groupby('estimated_gender')['cite_likelyhood'].sum())


# %% define filtered data slices
df_long_estd = df_long[df_long.estimated_gender != "Uncertain/Don't know"]
df_cm = df_long_estd[df_long_estd['egroup'] != 'TFemale']
df_cf = df_long_estd[df_long_estd['egroup'] != 'TMale']
df_fm = df_long_estd[df_long_estd['egroup'] != 'Control']

df_long_estd[df_long_estd['egroup'] == 'TFemale']['cite_likelyhood'].mean()
df_long_estd[df_long_estd['egroup'] == 'TMale']['cite_likelyhood'].mean()
df_long_estd[df_long_estd['egroup'] == 'Control']['cite_likelyhood'].mean()

df_long_estd[df_long_estd['egroup'] == 'TFemale']['cite_likelyhood'].std()
df_long_estd[df_long_estd['egroup'] == 'TMale']['cite_likelyhood'].std()
df_long_estd[df_long_estd['egroup'] == 'Control']['cite_likelyhood'].std()


# %% md
# # Make null model, random ensemble
# %% variables and filter to be used
filter_var = 'egroup'
filter_val_l = ['Control', 'TFemale', 'TMale']
stat_val = 'cite_likelyhood'

# # TODO: set up reproducible random generator
n = 1000
# rng = np.random.default_rng(seed=140923)
# a = rng.integers(low=100, high=10000, size=n)
# b = rng.integers(low=100, high=10000, size=n)

array_sum = np.zeros(6)
array_sum_est = np.zeros(6)

for i in range(n):
    # sampling from all responses to account for the whole system
    sample = df_long.sample(frac=.25)
    # sampling from all valid responses, participants who estimated gender
    sample_est = df_long_estd.sample(frac=.25)
    sample_count = Counter(sample[stat_val]).values()
    sample_count_est = Counter(sample_est[stat_val]).values()
    array_sum += list(sample_count)
    array_sum_est += list(sample_count_est)

# correct values for n-repetitions
Qk_null_count = array_sum/n
Qk_null_count_est = array_sum_est/n



# %% md
# # Compute some probabilities for validation
# %% # %% compute some manual probabilities for testing
Qk_count = Counter(df_long_estd[df_long_estd.egroup=='Control'][stat_val]).values()
Qk = [i/sum(Qk_count) for i in Qk_count]
Pk_count_m = Counter(df_long_estd[df_long_estd.egroup=='TMale'][stat_val]).values()
Pk_m = [i/sum(Pk_count_m) for i in Pk_count_m]
Pk_count_f = Counter(df_long_estd[df_long_estd.egroup=='TFemale'][stat_val]).values()
Pk_f = [i/sum(Pk_count_f) for i in Pk_count_f]


# Expected excess surprise from using Q as a model when the actual distribution is P
# Using the Control distribution as reference
sp.stats.entropy(Pk_m, Qk, base=2)
sp.stats.entropy(Pk_f, Qk, base=2)
# Using the Control ensemble as reference (null model)
sp.stats.entropy(Pk_m, Qk_null_count, base=2)
sp.stats.entropy(Pk_f, Qk_null_count, base=2)

a = [1,1,1]
b = [1,1,2]
sp.stats.entropy(a, b, base=2)


# %% md
# # Compare some quantities using statsmodels

# %% define response variable
y = df_long['cite_likelyhood']
#define predictor variables
x = df_long['egroup_n']
# add constant to predictor variables
x = sm.add_constant(x)
m = sm.OLS(list(Pk_count_m), list(Qk_count)).fit()
m.summary(), m.ssr

# fit linear regression model
mm = sm.OLS(list(Pk_count_m), list(Qk_count)).fit()
mm.summary(), mm.ssr

mf = sm.OLS(list(Pk_count_f), list(Qk_count)).fit()
mf.summary(), mf.ssr

# %% md
# # Information theory hypothesis testing
# %%
def calc_it_test(df, var, Qk_count, K):
    '''
    Calculate the relative entropy of given distributions.
    This routine will normalize pk and qk if they don’t sum to 1.
    Kullback motivated the statistic as an expected log likelihood ratio.
    AICc: the smaller the value the “closer” to full reality and is a good
    approximation for the information in the data, relative to the other models
    Compute
    '''

    labels_l = sorted(set(df.egroup))  # Get unique labels from the 'egroup' column and sort them
    Qk = [i/sum(Qk_count) for i in Qk_count]  # Normalize Qk_count to get probabilities
    temp = []
    for label in labels_l:
        Pk_count = Counter(df[df.egroup==label][var]).values()  # Get the count of values for the current label
        Pk = [i/sum(Pk_count) for i in Pk_count]  # Normalize Pk_count to get probabilities
        KL_D = sp.stats.entropy(Pk, Qk, base=2)  # Calculate the Kullback-Leibler divergence
        n = len(Qk)
        logL = -(n/2)*np.log2(KL_D/n)  # Calculate the log-likelihood
        AIC = -2*logL + 2*K  # Calculate the Akaike Information Criterion (AIC)
        AICc = AIC + (2*K*(K+1)) / (n-K-1)  # Calculate the corrected AIC (AICc)
        temp.append([label, KL_D, logL, AICc])  # Append the results to the temporary list
    df_it = pd.DataFrame(temp, columns=['label', 'KL_D', 'logL', 'AICc'])  # Create a DataFrame from the temporary list
    df_it['deltai'] = df_it['AICc'] - df_it['AICc'].min()  # Calculate the difference in AICc values
    df_it['wi'] = np.exp((-1/2)*df_it['deltai'])  # Calculate the weight of evidence
    df_it['wi'] = df_it['wi'] /df_it['wi'].sum()  # Normalize the weights
    return df_it  # Return the DataFrame with the calculated measures


# %% Qk_null includes all responses, use all data as reference distribution
calc_it_test(df_long_estd, 'cite_likelyhood', Qk_null_count, 2)

# %% has only estimated gender data as reference distribution
calc_it_test(df_long_estd, 'cite_likelyhood', Qk_null_count_est, 2)

# %% directly compare treatments with control
calc_it_test(df_fm, 'cite_likelyhood', Qk_count, 2)


# %%
# mean, std and quantiles,
# confidence interval


# %%
def print_it_test(df, var, Qk_count):
    '''
    Calculate the relative entropy of given distributions.
    This routine will normalize pk and qk if they don’t sum to 1.
    Kullback motivated the statistic as an expected log likelihood ratio.
    AICc: the smaller the value the “closer” to full reality and is a good
    approximation for the information in the data, relative to the other models
    Compute
    '''
    labels_l = sorted(set(df.egroup))
    Qk = [i/sum(Qk_count) for i in Qk_count]
    for label in labels_l:
        Pk_count = Counter(df[df.egroup==label][var]).values()
        Pk = [i/sum(Pk_count) for i in Pk_count]
        # log base 2
        KL_D = sp.stats.entropy(Pk, Qk, base=2)
        CE = sp.stats.entropy(Pk, base=2) + KL_D
        K = 2 # num parameters (variables)
        n = len(Qk)
        logL = np.log2(KL_D)
        AIC = -2*logL + 2*K
        AICc = AIC + (2*K*(K+1)) / (n-K-1)
        col = False
        if col:
            print(
                f"{'K-L divergence':<15} {label:>8}: {KL_D :.6f}",
                # f"\n{'Cross Entropy':<15} {P_label+'<>'+Q_label:>16}: {CE :.6f}",
                f"\n{'logL        ':<15} {label:>8}: {logL :.6f}",
                f"\n{'AICc        ':<15} {label:>8}: {AICc :.6f}"
                )
        else:
            print(f"| {label:<8} | {KL_D :.6f} | {logL :.6f} | {AICc :.6f} |")


# %%
print_it_test(df_fm, 'cite_likelyhood', Qk_count)

# %% md
# # Kullback–Leibler divergence between distribution
# %%
def print_kl_dist(df, var):
    '''
    Calculate the relative entropy of given distributions.
    If qk is not None, then compute the relative entropy
    D = sum(pk * log(pk / qk)). This routine will normalize pk and qk
    if they don’t sum to 1. Informally, the relative entropy quantifies the
    expected excess in surprise experienced if one believes the true
    distribution is qk when it is actually pk.
    Kullback motivated the statistic as an expected log likelihood ratio.
    AICc: the smaller the value the “closer” to full reality and is a good
    approximation for the information in the data, relative to the other models
    '''
    Q_label, P_label = sorted(set(df.egroup))
    Qk_count = Counter(df[df.egroup==Q_label][var]).values()
    Qk = [i/sum(Qk_count) for i in Qk_count]
    Pk_count = Counter(df[df.egroup==P_label][var]).values()
    Pk = [i/sum(Pk_count) for i in Pk_count]
    # log base 2
    KL_D = sp.stats.entropy(Pk, Qk, base=2)
    CE = sp.stats.entropy(Pk, base=2) + KL_D
    K = 2 # num parameters (variables)
    n = len(Qk)
    AIC = -2*np.log2(KL_D) + 2*K
    AICc = AIC + (2*K*(K+1)) / (n-K-1)
    print(
        f"{'K-L divergence':<15} {P_label+'<>'+Q_label:>16}: {KL_D :.6f}",
        f"\n{'Cross Entropy':<15} {P_label+'<>'+Q_label:>16}: {CE :.6f}",
        # f"\n{'AIC':<15} {P_label+'<>'+Q_label:>16}: {AIC :.6f}",
        f"\n{'AICc':<15} {P_label+'<>'+Q_label:>16}: {AICc :.6f}"
        )


# %% aggregated likelihoods
print_kl_dist(df_cm, 'sign_c')
print_kl_dist(df_cf, 'sign_c')
print_kl_dist(df_fm, 'sign_c')

# %% individual likelihoods
print_kl_dist(df_cm, 'cite_likelyhood_c')
print_kl_dist(df_cf, 'cite_likelyhood_c')
print_kl_dist(df_fm, 'cite_likelyhood_c')
