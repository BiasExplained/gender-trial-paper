# %%
import openai
import re

openai.api_key = open('openai_key', 'r').readline().strip()


def break_lines(text):
    # Check if the input text is a list, if not, convert it to a string
    if isinstance(text, list):
        list_words = text
    else:
        text = " ".join(text.split('\n\n'))
        list_words = text.split(" ")

    new_t = ""  # Initialize an empty string to store the formatted text
    line = ""  # Initialize an empty string to store the current line

    # Iterate through each word in the list of words
    for word in list_words:
        # If adding the current word to the current line does not exceed the character limit of 80
        if len(line) + len(word) <= 80:
            line += word + " "  # Add the word to the current line
        else:
            new_t += line + "\n"  # Add the current line to the formatted text with a line break
            line = word + " "  # Start a new line with the current word

    return new_t + line  # Return the formatted text with the last line


def get_keywords_abstract(context):
    try:
        response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Extract keywords from this scientific article abstract:\n\n{context}",
        temperature=0.3,
        max_tokens=20,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0
        )
        return response['choices'][0]['text']
    except:
        return ""


def get_questions_x(context):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=context,
            temperature=0.65,
            max_tokens=2500,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0
            )
        return response['choices'][0]['text']
    except:
        return ""


def call_openai_chat(prompt, temp, presence, freq):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that writes scientific research review articles."},
                {"role": "user", "content": prompt}
                ],
            temperature=temp,
            max_tokens=1500,
            top_p=1.0,
            presence_penalty=presence, #use [0.1, 1]
            frequency_penalty=freq #use [0.1, 1]
            )
        return response.to_dict()
    except:
        return response


def call_openai(prompt, temp, presence, freq):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temp,
            max_tokens=1500,
            top_p=1.0,
            presence_penalty=presence, #use [0.1, 1]
            frequency_penalty=freq #use [0.1, 1]
            )
        return response.to_dict()['choices'][0]
    except:
        return None


def get_title_and_abstract(raw_text):
    # case-insensitive pattern match
    has_pattern = re.search('abstract', raw_text, flags=re.IGNORECASE)
    if isinstance(has_pattern, re.Match):
        text = raw_text.replace('\n\n',' ').replace('\n',' ')
        start, end = has_pattern.span()
        # split text from match positions, 1 char shift
        title = text[: start-1].replace('Title','').strip('": ')
        abstract = text[end :]
    else:
        # replace weird linebreak
        raw_text = raw_text.replace('\n \n', '\n\n').replace('\n  \n', '\n\n')
        text_l = raw_text.strip().split('\n\n')
        n = len(text_l)
        # control for >1 linebreaks
        if n == 2:
            title = text_l[0]
            abstract = text_l[1]
        # elif n == 3:
            # title = text_l[:1]
            # abstract = text_l[-1]
        else:
            title = text_l[0]
            abstract = text_l[1:]
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    if isinstance(title, list):
        title = " ".join(title)
    return title, abstract.strip('": ')


def make_prompts(kw, a):
    t = 10 # number of words in title
    c1, c2 = kw[1], kw[0] # order by probability
    prompts = {
        "1a": f"Write a scientific journal abstract in {a} words for a scientific article about {c1} and {c2}:",
        "2a": f"Write an academic abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "2b": f"Write an academic journal abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "2c": f"Write a scientific journal abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "3a": f"Create a scientific journal abstract in {a} words for a scientific review article combining {c1} and {c2}:",
        "3b": f"Create a title and a scientific journal abstract in {a} words for a scientific review article combining {c1} and {c2}:",
        "3c": f"Create a title in {t} words and a scientific journal abstract in {a} words for a scientific review article combining {c1} and {c2}:",
        "4a": f"Create a scientific journal abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "4b": f"Create a title and a scientific journal abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "4c": f"Create a title in {t} words and a scientific journal abstract in {a} words for a scientific review article about {c1} and {c2}:",
        "5a": f"Create a title in {t} words and an abstract in {a} words for a scientific journal review article about {c1} and {c2}:",
        "5b": f"Create a scientific journal title in {t} words and a scientific journal abstract in {a} words for a scientific journal review article about {c1} and {c2}:",
        "6a": f"Create (1) a scientific research article title in {t} words and (2) an scientific research article abstract in {a} words for a scientific research review article about {c1} and {c2}:",
        "6b": f"Create (1) a title in {t} words and (2) an abstract in {a} words for a scientific research review article about {c1} and {c2}:",
        "6c": f"Create (1) a title and (2) an abstract in {a} words for a scientific research review article about {c1} and {c2}:",
        "6d": f"Create a title and an abstract in {a} words for a scientific research review article about {c1} and {c2}:",
        "6e": f'Create a title and an abstract in {a} words for a scientific research review article about: "{c1} and {c2}" '
        }
    return prompts
