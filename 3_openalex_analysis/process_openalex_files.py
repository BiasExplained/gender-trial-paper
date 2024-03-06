'''
This scrit processes openalex API response data and saves the result into a
single json file.
'''

import os
import tarfile
from glob import glob
import pandas as pd
import jsonlines as jsonl
import json
from tqdm import tqdm

home = os.path.expanduser('~')
data_path = "<path>/openalex_harvest"


# %%

def get_concepts(concept_list, level, th=0.0):
    # Function to get concepts for a given level and score
    temp = []
    # Check if the level is an integer or a list of integers
    if isinstance(level, int):
        level = [level]
    # Iterate over each level
    for lvl in level:
        # Iterate over each concept in the concept list
        for concept in concept_list:
            # Filter concepts based on level and score
            score = float(concept['score'])
            if (concept['level'] == lvl) and (score > th):
                name = concept['display_name']
                temp.append((name, score))
    # Sort the concepts based on score
    return sorted(temp, key=lambda i: i[1])



def get_all_concepts(concept_list, th=0.0):
    "Get concepts for all levels given a score"
    temp = []
    for concept in concept_list:
        # filter level
        score = float(concept['score'])
        if score > th:
            name = concept['display_name']
            level = concept['level']
            temp.append((name, score, level))
    return temp


def rebuild_abstract(words_dict):
    "Rebuild abstract from abstract_inverted_index"
    max_idx = max([max(i) for i in words_dict.values()]) + 1
    abstract = [None] * max_idx

    for word, pos in words_dict.items():
        for idx in pos:
            abstract[idx] = word

    return " ".join(abstract)


def calc_ratio_authors_orcid(doc):
    "Get number of authors per document and number of authors with and orcid"
    n_authors_orcid = [i for i in doc if i['author']['orcid'] is not None]
    return(len(doc), len(n_authors_orcid))


# %%
data_path = "<path>/res*.json"
files = glob(os.path.join(home, data_path))
concept_th = 0.5  # threshold for concept score - not used

header = [
        "openalex_id",
        "doi",
        "pub_date",
        "doc_type",
        "n_authors",
        "concept",
        "score",
        "level"]


save_path = "<path>"
filename = "processed_openalex_20220901-20230430.jsonl"

# Open the JSONL file for writing
with jsonl.open(os.path.join(save_path, filename), mode='w') as writer:
    # Write the header to the file
    writer.write(header)
    # Iterate over each file in the list of files
    for filename in tqdm(files):
        # Open the file for reading
        with open(filename, 'r') as f:
            # Load the JSON data from the file
            data = json.load(f)
            # Loop over each document in the 'results' field of the JSON data
            for doc in data['results']:
                # Get the authorships and number of authors for the document
                authorships = doc['authorships']
                n_authors = len(authorships)
                # Get the concepts for the document
                concepts = doc['concepts']
                # Skip documents with large collaborations or no concepts
                if (n_authors <= 25) and (len(concepts) > 0):
                    # Get the OpenAlex ID, publication date, and document type
                    openalex_id = doc['id'].split('/')[-1]
                    pub_date = doc['publication_date']
                    doc_type = doc['type']
                    # Get the DOI for the document if available
                    try:
                        doi = doc['doi']
                        doi = doi.replace('https://doi.org/', '')
                    except:
                        doi = ""
                    # Get all concepts for the document
                    clean_concepts = get_all_concepts(concepts)
                    # Write the results in long format (tidy) to the file
                    for values in clean_concepts:
                        concept, score, level = values
                        writer.write([
                            openalex_id,
                            doi,
                            pub_date,
                            doc_type,
                            n_authors,
                            concept,
                            score,
                            level
                        ])


# %%
savetofile = False
save_path = os.path.join(home, "<path>")
filename = "processed_openalex_20220901-20230430.jsonl"

if savetofile:
    with jsonl.open(os.path.join(save_path, filename), mode='w') as f:
        f.write_all(results)
    # alternative save method
    # df = pd.DataFrame(results, columns=['openalex_id','doi','pub_date','doc_type','n_authors','n_authors_orcid','concept','score','level'])
    # df.to_json(os.path.join(home, save_path, filename))

# %%
# load file for testing and validation
df = pd.read_json(os.path.join(save_path, filename), lines=True)
df = df.rename(columns=df.iloc[0]).drop(df.index[0])
df.head(3)
