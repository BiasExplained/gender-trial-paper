'''
This script creates multiple entries for the same email (doi-based)
Duplicates should be cleared afterwards, Qualtrics will overwrite the contact
data by the latest info and this behaviour can not be changed
'''

import os
import jsonlines as jsonl
import re
import random
import csv
from metaphone import doublemetaphone
from tqdm import tqdm
from collections import Counter
from glob import glob
from itertools import islice
from IPython.display import display
import jellyfish
import pandas as pd


data_path = os.path.join(os.path.expanduser('~'), "<path>")
pwd5 = "5_qualtrics_analysis/current"
pwd3 = "3_openalex_analysis/current"
pwd1 = "1_abstract_writing/abstracts"


# %%
def dmetaphone(tp1, tp2, threshold):
    tuple1 = doublemetaphone(tp1)
    tuple2 = doublemetaphone(tp2)
    if threshold == 0:
        if tuple1[1] == tuple2[1]:
            return True
    elif threshold == 1:
        if tuple1[0] == tuple2[1] or tuple1[1] == tuple2[0]:
            return True
    else:
        if tuple1[0] == tuple2[0]:
            return True
    return False


def split_contact_text(author_txt):
    i = author_txt.lower().find('corr')
    if i >= 0:
        author_txt = author_txt[: i-1].strip()
    if ',' in author_txt and ' ' in author_txt:
        last_name, first_name = re.split(",", author_txt, maxsplit=1)
    elif ',' in author_txt and ' ' not in author_txt:
        last_name, first_name = re.split(",", author_txt, maxsplit=1)
    elif ',' not in author_txt and ' ' in author_txt:
        first_name, last_name = re.split("\\s", author_txt, maxsplit=1)
    else:
        first_name, last_name = ('', author_txt)
        # print('No last name: ', author_txt)
    return first_name.strip(), last_name.strip()


def clean_text(text):
    # split multiple words
    text = re.sub(r'[\(\)]', '', text)
    parts = re.split("[-,.\\s]", text)
    # remove single letter
    parts = [i.lower().strip() for i in parts if len(i.replace('.', '')) > 1]
    return parts


def pattern_in_alias(surname_name, alias, strict=True):
    last_name, first_name = split_contact_text(surname_name)
    surnames = clean_text(last_name)
    # try to match surname units
    match_surnames_l = [re.search(s, alias, flags=re.IGNORECASE) for s in surnames]
    check = len([i for i in match_surnames_l if isinstance(i, re.Match)])
    if check == 0:
        names = clean_text(first_name)
        # try to match name units
        match_names_l = [re.search(s, alias, flags=re.IGNORECASE) for s in names]
        check = len([i for i in match_names_l if isinstance(i, re.Match)])
        if check == 0:
            # try to match surname units by tone
            match_surname_tone = [dmetaphone(i, alias, 2) for i in surnames]
            check = len([i for i in match_surname_tone if i is True])
            if check == 0:
                # try to match surname units with typo
                dl_dist = [jellyfish.damerau_levenshtein_distance(i, alias) <= 1 for i in surnames]
                check = len([i for i in dl_dist if i is True])
                # Only try pieces if no strict criteria
                if check == 0 and strict is False:
                    pieces1 = [i[: len(i)//2+1] if len(i) > 4 else i for i in surnames+names]
                    pieces2 = [i[len(i)//2+1:] if len(i) > 4 else i for i in surnames+names]
                    # split again long names + surnames > 4 chars
                    pieces3 = [i[:4] if len(i) > 4 else i for i in pieces1+pieces2]
                    # take only pieces > 2, 2-char common in asian names
                    pieces = set([i for i in pieces1+pieces2+pieces3 if len(i) > 2])
                    match_pieces_l = [re.search(p, alias, flags=re.IGNORECASE) for p in pieces]
                    check = len([i for i in match_pieces_l if isinstance(i, re.Match)])
    return bool(check > 0)


cleaner = re.compile('<.*?>')


# %% Load wos data, can be provided upoen request
fname_wos = "processed_wosdata_20220901-20230430_titles-fixed.jsonl"
wos_dic = {}

with jsonl.open(os.path.join(data_path, pwd3, fname_wos)) as r:
    r.read()
    for obj in tqdm(r):
        doi = obj[4]
        # if doi in filter:
        wos_dic[doi] = {
            'authors': obj[0],
            'doctype': obj[1],
            'contact_text': obj[2],
            'contact_email': obj[3],
            'wos_id': obj[5],
            'date': obj[6],
            'title': obj[7]
            }

# %%
# dfv = pd.DataFrame(wos_dic)

# %% Load abstracts
pattern_abs = "openai_response_multistep_novel*"
abstracts_files = glob(os.path.join(data_path, pwd1, pattern_abs))

abstracts_dic = {}
for file in abstracts_files:
    with jsonl.open(os.path.join(data_path, pwd1, file)) as r:
        for obj in r:
            w1, w2 = obj['keywords']
            abstracts_dic[(w1, w2)] = {
                'title': obj['title'].strip('" '),
                'abstract': obj['abstract'].strip('" '),
                'keywords': obj['keywords'][0]
                }


# %% load keywords from dic, as in write_abstract script:
# USE A FILE WITH ALL KW-DOI DIC TO AVOID COUNTING DUPLICATES
# File can be created by the process_pickle_files.py script
kw_multistep_file = "dic_kw_multistep_novel_doi_20220901-20230430.jsonl"

doi_kw_dic = {}
with jsonl.open(os.path.join(data_path, pwd3, kw_multistep_file)) as r:
    for obj in r:
        doi_kw_dic[tuple(obj[0])] = obj[1]  # list of DOIs per KW


# %% load contacted emails to filter out
# File not provided for privacy reasons, code left for auditing purposes
contacted_emails = set()
pattern = "contacted_emails_qualtrics_wos*"
contacted_emails_files = glob(os.path.join(data_path, pwd5, pattern))

for file in contacted_emails_files:
    with open(file) as f:
        for obj in f:
            contacted_emails.add(obj.lower().strip())


# %% define fileds and experiment data
csv_fields = [
    'email',
    'title_abstract',
    'abstract_1',
    'abstract_2',
    'female_author_first_name',
    'male_author_first_name',
    'control_author_first_name',
    'author_last_name',
    'target_author_first_name',
    'target_author_last_name',
    'target_author_title_abstract',
    'keywords',
    'doi'
    ]

authors_dic = {
    'Brown':    {'F': 'Jennifer',   'M': 'James',     'C': 'E.'},
    'Miller':   {'F': 'Julie',      'M': 'John',      'C': 'J.'},
    'Moore':    {'F': 'Madeline',   'M': 'Mark',      'C': 'M.'},
    'Thompson': {'F': 'Mary',       'M': 'Mathew',    'C': 'A.'},
    'White':    {'F': 'Michelle',   'M': 'Michael',   'C': 'C.'},
    'Walker':   {'F': 'Natalie',    'M': 'Nicholas',  'C': 'N.'},
    'Young':    {'F': 'Samantha',   'M': 'Stephen',   'C': 'S.'},
    'Hill':     {'F': 'Ann',        'M': 'Noah',      'C': 'O.'},
    'King':     {'F': 'Caroline',   'M': 'Daniel',    'C': 'L.'},
    'Wright':   {'F': 'Emily',      'M': 'Paul',      'C': 'G.'},
    }


# %% Start the main file build
cleared_doi = []
# processed_emails = set()
n = 0
result = []
unmatched = []

# loop over GPT-4 generated abstracts
# for kw_pair, abs_data in list(tqdm(abstracts_dic.items()))[:1]:
for kw_pair, abs_data in tqdm(abstracts_dic.items()):
    # loop over dictionary with DOIs for a kw pair
    # result = []
    doi_list = doi_kw_dic.get(kw_pair)
    if doi_list is None:
        # check the reversed kw_pair
        doi_list = doi_kw_dic.get(kw_pair[::-1])
    for doi in doi_list:  #
        n += 1  # count processed DOIs
        # get data from WOS (real papers)
        wosdata = wos_dic.get(doi)
        if wosdata is None or doi in cleared_doi:
            continue

        authors = wosdata['authors'].split(';')
        contact_email = wosdata['contact_email'].split(';')
        contact_text = wosdata['contact_text'].split(';')
        contact_text = [i for i in contact_text if len(i) > 1]
        # always get the first email
        email = contact_email[0].lower().strip()
        # check if doi already included or email contacted, skip if so
        if email in contacted_emails:  # or email in processed_emails:
            continue

        # Get the corresponding author name and surname
        # Single author or corresponding author
        if len(authors) == 1:
            target_author_first_name, target_author_last_name = split_contact_text(authors[0])
        elif len(contact_email) == 1 and len(contact_text) > 0:
            target_author_first_name, target_author_last_name = split_contact_text(contact_text[0])
        # Several authors
        else:
            # Try to match first email
            email_alias = email.split('@')[0]  # alias only
            match_authors = [pattern_in_alias(a, email_alias) for a in authors]
            idx = [i for (i, var) in enumerate(match_authors) if var is True]
            if len(idx) == 0:
                # Try to match second email if exists
                if len(contact_email) > 1:
                    email = contact_email[1].lower().strip()
                    email_alias = email.split('@')[0]  # alias only
                    match_authors = [pattern_in_alias(a, email_alias) for a in authors]
                    idx = [i for (i, var) in enumerate(match_authors) if var is True]
                    # stops if it fails
                    if len(idx) == 0:
                        unmatched.append([email, authors, contact_email, contact_text, doi])
                        continue
                    else:
                        target_author_first_name, target_author_last_name = split_contact_text(authors[idx[0]])
                else:
                    unmatched.append([email, authors, contact_email, contact_text, doi])
                    continue
            else:
                target_author_first_name, target_author_last_name = split_contact_text(authors[idx[0]])
        # if len(target_author_first_name) == 0 and len(target_author_last_name) == 0:
        #     print(n, target_author_first_name, target_author_last_name, email, authors, contact_email, contact_text, doi)
        #     continue

        # Prepare metadata
        title_abstract = abs_data['title']
        #  split abstract to avoid 1024 char limit by qualtrics
        if len(abs_data['abstract']) <= 1020:
            abstract_1 = abs_data['abstract']
            abstract_2 = ''
        else:
            abstract_1 = abs_data['abstract'][: 1000]
            abstract_2 = abs_data['abstract'][1000:]
        # get other data
        author_last_name = random.choice(list(authors_dic.keys()))
        female_author_first_name = authors_dic[author_last_name]['F']
        male_author_first_name = authors_dic[author_last_name]['M']
        control_author_first_name = authors_dic[author_last_name]['C']
        target_author_title_abstract = re.sub(cleaner, '', wosdata['title'])
        kw_str = "_".join(kw_pair).replace(' ', '-').replace('/', '-')
        # append results using qualtrics format
        result.append([
                email,
                title_abstract,
                abstract_1,
                abstract_2,
                female_author_first_name,
                male_author_first_name,
                control_author_first_name,
                author_last_name,
                target_author_first_name.strip(),
                target_author_last_name.strip(),
                target_author_title_abstract,
                kw_str,
                doi
                ])
        cleared_doi.append(doi)


# %%
# wos_authors: 19229
# wos_authors_283193: 245868
# total: 265,097
run = True
if run:
    print(len(result))  # 345334
    print(len(unmatched))  # 4979
    dff = pd.DataFrame(result, columns=csv_fields)
    dff.doi.nunique()  # 345334
    dff.email.nunique()  # 289677
    len(abstracts_dic)  # 417
    dff.head()


# %% save everything to a single file
savefile = True
if savefile:
    # posfix = '20230101-20230430'
    posfix = 'write-test'
    fname_wos_aut = f'qualtrics_contacts_wos_{len(result)}_multistep_novel_{posfix}_newsurnames_tab.csv'
    with open(os.path.join(data_path, pwd5, fname_wos_aut), "w", newline='', encoding="utf-8-sig") as f:
        spamwriter = csv.writer(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(csv_fields)
        # make dataframe and group by email
        df = pd.DataFrame(result, columns=csv_fields)
        # make groups by email to select with paper for contact subject
        groups = df.groupby('email')
        n = 0
        # loop over groups
        for name, group in groups:
            if group.shape[0] > 1:
                # if more than one doc per email, select based on order below
                for i, row in group.iterrows():
                    v = wos_dic.get(row['doi'])['doctype']
                    if isinstance(re.search(v, 'Article', flags=re.IGNORECASE), re.Match):
                        spamwriter.writerow(row.tolist())
                        break
                    elif isinstance(re.search(v, 'Letter', flags=re.IGNORECASE), re.Match):
                        spamwriter.writerow(row.tolist())
                        break
                    elif isinstance(re.search(v, 'Proceedings', flags=re.IGNORECASE), re.Match):
                        spamwriter.writerow(row.tolist())
                        break
                    else:
                        spamwriter.writerow(row.tolist())
                        break
            else:
                spamwriter.writerow(group.values.flatten().tolist())
            n += 1
    # plus heading
    print(f'{n} rows saved to file!\n',
          f'{df.email.nunique()} unique emails in dataframe\n',
          f'{len(groups)} groups'
          )

# %% check data
# Make some validations
run = False
if run:
    display(list(islice(abstracts_dic.items(), 6)))
    display(dict(list(abstracts_dic.items())[0:2]))

    sorted(Counter([i[0] for i in result]).items(), key=lambda x: x[1])

# %%
for line in result[:1000]:
    temail = line[0]
    tname = line[8]
    tsurname = line[9]
    if tname.lower() not in temail or tsurname.lower() not in temail:
        print(tname, tsurname, line[0], temail)
