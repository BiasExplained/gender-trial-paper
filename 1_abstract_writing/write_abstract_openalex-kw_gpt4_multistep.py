import os
import textstat
import openai
import time
import re
import jsonlines as jsonl
import json
from utils import break_lines, make_prompts, get_title_and_abstract
from tqdm import tqdm
from glob import glob
from tenacity import (retry, stop_after_attempt, wait_random_exponential)


home = os.path.join(os.path.expanduser('~'), "<path>")
pwd1 = "data/abstracts"

# %%
openai.api_key = open('openai_key', 'r').readline().strip()


# Retry decorator for exponential backoff
@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    try:
        return openai.ChatCompletion.create(**kwargs)
    except Exception as e:
        print(e)
        raise e


def call_openai_chat(prompt, n_words, temp, presence, freq):
    try:
        kw1, kw2 = prompt
        res = completion_with_backoff(
            model="gpt-4",
            messages = [
            {'role': 'system', 'content': "Your tasks is to write a scientific research article."},
            {'role': 'user', 'content': f"Create a title and an abstract in {n_words} words for a scientific research article about `{kw1}` and `{kw2}`, including description of a novel hypothesis about their connection."},
            {'role': 'user', 'content': f"Generate 2 rounds of feedback and improvement using the following prompt: `Provide a critical evaluation of this abstract, focusing on the degree to which it proposes a specific and novel hypothesis, then rewrite the title and abstract in {n_words} words to make them more compelling following this evaluation.`"},
            {'role': 'user', 'content': "Select the last abstract from this sequential exercise."},
            {'role': 'user', 'content': "Output the final title and abstract as a JSON object containing the keys: title, abstract."}
            ],
            temperature=temp,
            max_tokens=1500,
            top_p=1.0,
            presence_penalty=presence, #use [0.1, 1]
            frequency_penalty=freq #use [0.1, 1]
            )
        return res['choices'][0]['message']['content']
    except:
        return res


def response_quality(text, n):
    '''
    Check test lenght only, can be adjusted to test patter ("review|overview" in text)
    '''
    # only abstracts containing review|overview
    has_pattern = re.search('review|overview', text, flags=re.IGNORECASE)
    type_test = isinstance(has_pattern, re.Match)
    # only abstracts between n-20 and n+50 words
    len_test = n+50 >= len(text.split()) >= n-10
    return True, len_test


# %% load kw_pairs with abstracts already created
kwpairs_with_abstract = set()
pattern = "openai_response_multistep_*"
kwpairs_abstract_files =  glob(os.path.join(home, pwd1, pattern))

for file in kwpairs_abstract_files:
    with jsonl.open(file) as f:
        for obj in f:
            kwpair = tuple(obj['keywords'])
            kwpairs_with_abstract.add(kwpair)

# %%
kwords_l = []
if len(kwords_l) == 0:
    # load keywords from dic,
    # USE A FILE WITH ALL KW-DOI DIC TO AVOID COUNTING DUPLICATES
    kw_multistep_file = "dic_kw_multistep_novel_doi_20220901-20230430.jsonl"
    dic_kw = {}
    with jsonl.open(os.path.join(home, pwd1, kw_multistep_file)) as f:
        for obj in f:
            dic_kw[tuple(obj[0])] = len(obj[1]) # number DOIs per KW
    th = 400
    # Select only kw with frequency > th and no abstract created
    kwords_l = [k for k,v in dic_kw.items() if v > th and k not in kwpairs_with_abstract]
    print(len(kwords_l))


# %%
# initialise parameters space
n_words = 150
prompt_id = 'ms'
temperature = 0.7
presence = 0
frequency = 0
responses_l = []

# Loop through each keyword in kwords_l
for kwords in tqdm(kwords_l):
    # Call the function call_openai_chat with the specified parameters
    text = call_openai_chat(kwords, n_words, temperature, presence, frequency)
    # Calculate the number of times the response passes the response_quality check
    passed = sum(response_quality(text, n_words))
    # Try 3 times to get a response containing a valid abstract
    attempts = 0
    while passed <= 1:
        # Call the function call_openai_chat again to get a new response
        text = call_openai_chat(kwords, n_words, temperature, presence, frequency)
        # Recalculate the number of times the response passes the response_quality check
        passed = sum(response_quality(text, n_words))
        attempts += 1
        time.sleep(0.1)
        if attempts == 3:
            # If the maximum number of attempts is reached, set text to False and print a failure message
            text = False
            print('Failed: ', f"{temperature}_{presence}_{frequency}", kwords)
            break
    if text is not False:
        # Create a dictionary to store the response details
        res = {}
        res['iter'] = f"{temperature}_{presence}_{frequency}"
        res['keywords'] = kwords
        res['prompt'] = prompt_id
        res['raw'] = text
        res['temperature'] = temperature
        res['presence'] = presence
        res['frequency'] = frequency
        try:
            # Parse the response text as JSON and extract the title and abstract
            text_json = json.loads(text)
            res['title'] = text_json['title']
            res['abstract'] = text_json['abstract']
        except:
            pass
        # Append the response dictionary to the responses_l list
        responses_l.append(res)


# %%
# print(break_lines(res['title']))
# print(break_lines(res['abstract']))


# %%
# save responses to a jsonl file
if len(kwords_l) == 1:
    posfix = "_".join(kwords).replace(' ','-')
    fname_out = f"openai_response_multistep_novel_{posfix}.jsonl"
else:
    version = "multistep_novel"
    posfix = "3"
    fname_out = f"openai_response_{version}_{posfix}.jsonl"

with jsonl.open(os.path.join(home, pwd1, fname_out), mode='w') as f:
    f.write_all(responses_l)
