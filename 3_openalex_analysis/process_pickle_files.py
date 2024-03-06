"""
This script processes pickle files containing keyword-concept mappings and performs keyword pair matching based on certain criteria. It also merges data from WOS (Web of Science) and OpenAlex datasets.

The script performs the following steps:
1. Reads the pickle files containing keyword-concept mappings.
2. Inverts the mappings to create dictionaries with DOI-based and shortDOI-based keys.
3. Reads the merged WOS-OA (Web of Science-OpenAlex) data from a JSONL file.
4. Groups the data by DOI and extracts the keywords and scores from OpenAlex.
5. Matches the BERT (Bidirectional Encoder Representations from Transformers) keywords with the merged data keywords.
6. Filters and processes the keyword pairs based on certain conditions.
7. Updates a dictionary with the selected keyword pairs and their respective DOIs.
8. Prints the statistics and saves the selected keyword pairs to a JSONL file.
9. Performs additional tests and validations.

Note: The script contains multiple sections marked by '%%' which are used for code organization and execution in Jupyter Notebook.
"""

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import jsonlines as jsonl
from itertools import combinations, product, groupby, islice
from operator import itemgetter
from collections import Counter

wkd6 = "../data/abstracts"
wkd3 = "../data/keywords_analysis"


# %%
# Define a function to update a keyword dictionary with a keyword pair and its corresponding DOI
def update_kw_dic(kw_dic, pair, doi):
    if pair not in kw_dic:  # If the keyword pair is not already in the dictionary
        pair_inverted = pair[::-1]  # Invert the order of the keyword pair
        if pair_inverted in kw_dic:  # If the inverted keyword pair is already in the dictionary
            kw_dic[pair_inverted].append(doi)  # Append the DOI to the inverted keyword pair
        else:  # If the inverted keyword pair is not in the dictionary
            kw_dic[pair] = [doi]  # Add the keyword pair to the dictionary with the DOI as a list
    else:  # If the keyword pair is already in the dictionary
        kw_dic[pair].append(doi)  # Append the DOI to the existing keyword pair in the dictionary


# %%
# BERT process file matching concepts to paper dois, where papers are matched
# to the concepts by OpenAlex. Format: {concept: [doi1, doi2, …]}

# Define the pattern for concept-doi pickle files
pattern_concept_doi = "concept*doi.pkl"

# Get the list of concept-doi files using the pattern
concept_doi_files = glob(os.path.join(wkd3, pattern_concept_doi))

# Create dictionaries to store the inverted mappings
doi_concept_d = {}  # Inverted dictionary with DOI-based keys
doi_part_d = {}  # Inverted dictionary with shortDOI-based keys

# Iterate over each concept-doi file
for file in concept_doi_files:
    # Read the concept-doi pickle file
    concept_doi = pd.read_pickle(file)

    # Iterate over each keyword and DOI pair in the concept-doi dictionary
    for kw, doi_l in tqdm(concept_doi.items()):
        # Iterate over each DOI in the DOI list
        for doi in doi_l:
            # Remove the 'https://doi.org/' prefix from the DOI
            doi_s = doi.replace('https://doi.org/', '')

            # Split the DOI to get the shortDOI part
            doi_part_d[doi_s.split('/')[1]] = doi_s

            # Update the doi_concept_d dictionary with the keyword and DOI pair
            if doi_s in doi_concept_d:
                doi_concept_d[doi_s].append(kw)
            else:
                doi_concept_d[doi_s] = [kw]

# %%
# BERT process file Keyword pairs filtered from above, only keyword pairs
# both of which are in 1 are retained. Format:
#     [(keyword1-1, keyword1-2), (keyword2-1, keyword2-2), …]
# Note: N=74303, 84701, 86531, 80351 for each list

kw_filtered_file = 'filter_q1_1e-4.pkl'
kw_filtered = pd.read_pickle(os.path.join(wkd3, kw_filtered_file,))
kw_filtered_s = set(kw_filtered)
# kw_set = set(map(itemgetter(0), kw_filtered)) | set(map(itemgetter(1), kw_filtered))

# dic with a keyword and all their pairs
kw_filtered_dic = {}
for i, j in kw_filtered_s:
    if i in kw_filtered_dic:
        kw_filtered_dic[i].append(j)
    if i not in kw_filtered_dic:
        kw_filtered_dic[i] = [j]
    if j in kw_filtered_dic:
        kw_filtered_dic[j].append(i)
    if j not in kw_filtered_dic:
        kw_filtered_dic[j] = [i]

# %% check some examples
# [i for i in kw_filtered_s if 'adaptive' in (i[0]+i[1]).lower()]
# next(iter(kw_filtered_dic))
# list(kw_filtered_dic.items())[:10]


# %% load merged WOS and OnpenAlex data
# Can be made available upon request
merge_file = "merged_wos-oa_20220901-20230430.jsonl"
merged = []
filter = ''
with jsonl.open(os.path.join(home, prj, wkd3, merge_file)) as reader:
    header = reader.read()
    if filter == '':
        for obj in tqdm(reader):
            merged.append(obj)
    else:
        date = obj[2]
        # yearmonth = "-".join(date.split('-')[:2])
        year = date.split('-')[0]
        if year == filter:
            merged.append(obj)


# %% set up temporary list
doi_error = []
failed = []
kwpairs_selected = {}
p, s = 0, 0
ps_counter = {1: 0, 2: 0}

# group all keyword with respective score by DOI using merged WOS-OA data
# 1 sets the index of the attrivute to group by in the list (DOI)
for doi, group in tqdm(groupby(merged, itemgetter(1))):
    # get the keywords and scores from OpenAlex, by index
    mconcept_score = {i[5]: i[6] for i in group}

    try:
        # get keywords associated to a DOI matched by the BERT method
        processed_concepts = {i: 0 for i in doi_concept_d.get(doi)}
        for pconcept in processed_concepts:
            for mconcept, mscore in mconcept_score.items():
                # check if BERT keyword matches the merged data keyword
                # Some keywords were different due to cleaning methods for BERT
                if pconcept == mconcept or pconcept in mconcept:
                    # update processed_concepts with the score from OpenAlex
                    # Keep BERT keyword naming
                    processed_concepts[pconcept] = mscore
    except TypeError:
        doi_error.append(doi)
        continue

    # Process the keywords
    pairs_l = []
    # Handle cases with several keywords for a DOI, from BERT
    if len(processed_concepts) >= 2:
        # print("pc > 2")
        # get all combinations and make pairs
        pconcepts_combine = list(combinations(processed_concepts, 2))
        for kw1, kw2 in pconcepts_combine:
            # get all pairs for kw1 from BERT filtered pairs
            kw1_elems = kw_filtered_dic.get(kw1)
            # if the pair exists in BERT filtered pairs
            if kw1_elems is not None and kw2 in kw1_elems:
                # get the score of the pair and add to pair list
                score_pair = processed_concepts[kw1] + processed_concepts[kw2]
                pairs_l.append(((kw1, kw2), score_pair, 2))
                p += 1

    # Handle cases with a single keyword match or failed pair search
    if len(processed_concepts) == 1 or len(pairs_l) == 0:
        # print("pc = 1 or pairs = 0")
        for kw in processed_concepts.keys():
            # get all pairs for kw1 from BERT filtered pairs
            kw_elems = kw_filtered_dic.get(kw)
            if kw_elems is not None:
                score_pair = processed_concepts[kw]
                # Pair the kw with a keyword with the highest score
                pairs_l.append(((kw, sorted(kw_elems)[0]), score_pair, 1))
                s += 1

    if len(pairs_l) > 0:
        # print("pairs > 0")
        # sort list of pairs and select the highest score
        pairs_l.sort(key=itemgetter(1), reverse=True)
        # get pair with highest score with respective DOI
        update_kw_dic(kwpairs_selected, pairs_l[0][0], doi)
        ct = Counter((i[2] for i in pairs_l))
        for k, v in ct.items():
            ps_counter[k] += v
    else:
        failed.append(doi)


# %% Print same values for validation
print(len(merged))
print(len(set(i[1] for i in merged)))
print(len(doi_error))
print(len(failed))
print(len(kwpairs_selected))
print(p, s, p/s)
print(ps_counter)

# %% Save file will all KW
run = False
if run:
    fname_out = "dic_kw_multistep_novel_doi_20220901-20230430.jsonl"
    with jsonl.open(os.path.join(home, prj, wkd3, fname_out), mode='w') as f:
        f.write_all(kwpairs_selected.items())


# %% md --
# # Make tests and validate procedure
# %%
run = False
if run:
    print(len(set((i[1] for i in merged))))
    print(len(set(doi_error)))
    print(len(set(failed)))
    print(len(set(kwpairs_selected)))

    counter = {i: len(j) for i, j in kwpairs_selected.items()}
    sorted(counter.items(), key=itemgetter(1), reverse=True)
    sorted(counter.items(), key=itemgetter(1))


# %% Check for DOI mismatch by doing a partial search
# Already checked for 2022-09 to 2023-04 data, NO match found
run = False
if run:
    doi_part_s = set(doi_part_d.keys())
    for d in tqdm(doi_error):
        d1 = "".join(d.split('/')[1:]).strip()
        if d1 in doi_part_s:
            print(f"{d} - {d1} - {doi_part_d.get(d1)}")


# %% Check for keywork mismatch by doing a partial search
run = False
if run:
    i = failed[0]
    j = doi_concept_d.get(i)[0]
    print(i, j)
    display(kw_filtered_dic.get(j))
    gn = (k for k in kw_filtered_dic.keys() if j.split()[0] in k or j.split()[1] in k)
    display(list(gn))
    display(list(islice(mconcept_score.items(), 6)))
    # find example for specific kw pair
    set(concept_doi.get("Gene expression")).intersection(set(concept_doi.get("Adaptive evolution")))
    display(list(islice(concept_doi.items(), 1)))
