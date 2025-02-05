# %% codecell
# Import necessary libraries
import pandas as pd
import os

# Define home directory and project working directory
prjwd = "../data"

# Flag to save figures and path to save plots
savefig = True
plots_path = "plots_likert"


# %% markdown
# # Clean responses and make some checks
# %%
# Define categorical scores for Likert scale responses
cat_scores = {
    'Strongly unlikely': -3,
    'Quite unlikely': -2,
    'Somewhat unlikely': -1,
    'Somewhat likely': 1,
    'Quite likely': 2,
    'Strongly likely': 3
}

# Define aggregated scores for Likert scale responses
agg_scores = {
    'Unlikely': -1,
    'Likely': 1,
}

# Define credibility scores for responses
cred_scores = {
    'Not credible': -2,
    'Somewhat credible': -1,
    'Quite credible': 1,
    'Very credible': 2
}

# Define aggregated credibility scores
agg_cred = {
    'Not credible': -1,
    'Credible': 1,
}

# Define experimental groups
egroup_l = ['TFemale', 'Control', 'TMale']

# %% codecell
# Load the raw data from the specified CSV file
fname = "sciofsci_double-effect_-_Live_April_15,_2024_07.14_anonymised.csv"
raw = pd.read_csv(os.path.join(prjwd, fname), header=0)


# --------------------------------------------------------------------------
# %% Create a smaller copy
df = raw[[
    'ResponseId', 'Progress', 'credibility', 'estimated_gender_reason',
    'estimated_gender', 'occupation', 'participant_gender',
    'TFemale', 'Control', 'TMale', 'female_author_first_name',
    'male_author_first_name', 'control_author_first_name', 'keywords'
    ]].copy()

# Create a cite_likelihood column by consolidating the 3 groups
df['cite_likelihood'] = df[egroup_l].fillna('').sum(axis=1)

# Create an egroup column based on which group a response went
df['egroup'] = ''
df.columns.get_loc('egroup')
for i in df.iterrows():
    idx = df.columns.get_loc('egroup')
    if not pd.isna(i[1].TFemale):
        df.iloc[i[0], idx] = 'TFemale'
    elif not pd.isna(i[1].TMale):
        df.iloc[i[0], idx] = 'TMale'
    elif not pd.isna(i[1].Control):
        df.iloc[i[0], idx] = 'Control'
    else:
        pass

# Create numeric cite_likelihood column (validated)
df['cite_likelihood_num'] = df['cite_likelihood'].apply(
    lambda x: cat_scores.get(x))
df['cite_likelihood_num'] = df['cite_likelihood_num'].astype(
    "int", errors='ignore')

# drop responses where first question was not answered
df.drop(df[df.cite_likelihood == ''].index, inplace=True)

# Create numeric credibility column (validated)
df['credibility_num'] = df['credibility'].apply(lambda x: cred_scores.get(x))
df['credibility_num'] = df['credibility_num'].astype("int", errors='ignore')

# Create binart sign for cite_likelihood (validated)
bins = [-4, 0, 3]
names = ['Unlikely', 'Likely']
df['sign'] = pd.cut(df['cite_likelihood_num'], bins, labels=names)
df['sign_num'] = df['sign'].apply(lambda x: agg_scores.get(x))

# Create binart sign for credibility (validated)
bins2 = [-3, 0, 2]
names2 = ['Not credible', 'Credible']
df['credibility_sign'] = pd.cut(df['credibility_num'], bins2, labels=names2)
df['credibility_sign_num'] = df['credibility_sign'].apply(
    lambda x: agg_cred.get(x))

# set data types
df['occupation'] = df['occupation'].astype('category')
df['egroup'] = df['egroup'].astype('category')
df['participant_gender'] = df['participant_gender'].astype('category')
df['estimated_gender'] = df['estimated_gender'].astype('category')
df['cite_likelihood'] = pd.Categorical(
    df['cite_likelihood'], categories=cat_scores.keys(), ordered=True)
df['credibility'] = pd.Categorical(
    df['credibility'], categories=cred_scores.keys(), ordered=True)

# remove duplicated columns, now in egroup
df.drop(columns=['TFemale', 'Control', 'TMale'], inplace=True)
