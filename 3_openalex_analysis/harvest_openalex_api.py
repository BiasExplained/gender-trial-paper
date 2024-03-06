'''
This scipt access the openalex API and extract papers within a specific
time window

Work object:
    [id, doi, title, publication_date]
    concepts:
    is_paratext:False to remove random stuff
    author_position: are first, middle, and last.
    abstract_inverted_index: needs reconstruction
    authorships
Author object:
    [id, orcid, display_name]
    last_known_institution
Concept object
    [id, display_name, ]
    counts_by_year: peak concepts with high frequency
    ancestors: where that concept descends from

OpenAlex ID: W(ork), A(uthor), V(enue), I(nstitution),|C(oncept)
negation:!, OR:|
filters: https://docs.openalex.org/api/get-lists-of-entities/filter-entity-lists
'''

import requests
import json
import datetime
import time
import pandas as pd
import os
from tenacity import (retry, stop_after_attempt, wait_exponential)


# %%
# Retry decorator with exponential backoff
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(10))
def call_openalex_with_backoff(**kwargs):
    try:
        return requests.get(**kwargs)  # Send GET request to the specified URL
    except Exception as e:
        print(e)  # Print any exception that occurs during the request


# %%

# Set the email address for API requests
mailto = "mailto=<your-email-address>"

# Set the number of results per page
per_page = 200

# Set the start and end dates for the time window
from_date = "2022-09-01"
to_date = "2023-04-30"

# Create the period filter for the API request
period = f"from_publication_date:{from_date},to_publication_date:{to_date}"

# Set the sorting criteria for the API request
sort_by = "publication_date"

# Set the base URL for the API
base_url = "https://api.openalex.org/"

# Set the initial cursor value
cursor = "*"

# Construct the API query URL
query = f"{base_url}works?filter=is_paratext:false,{period}&per-page={per_page}&{mailto}&cursor={cursor}"

# Set the output path for the harvested data
out_path = "<path>"

# Set the timeout for API requests
time_out = 2

# Set the sequence number for the harvested data files
seq = 0

# %%
# Start a loop to retrieve data from the API using pagination
while cursor is not None:
    # Call the API with exponential backoff
    response = call_openalex_with_backoff(url=query, timeout=time_out)
    try:
        response_j = response.json()
        # Check if the response contains results
        if len(response_j['results']) > 0:
            # Save the response as a JSON file
            with open(f"{out_path}response_openalex_{seq}.json", "w", encoding='utf8') as file:
                json.dump(response_j, file, ensure_ascii=False)
    except:
        pass
    # Update the cursor for the next API request
    cursor = response_j['meta']['next_cursor']
    query = f"{base_url}works?filter=is_paratext:false,{period}&per-page={per_page}&{mailto}&cursor={cursor}"
    seq += 1
    print(response_j['meta']['count'])
    # Wait for a short period before making the next API request
    time.sleep(0.5)
