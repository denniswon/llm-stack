import os
import pandas as pd
import openai

import chromadb
from chromadb.utils import embedding_functions

import tiktoken
from scipy import spatial

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Step 1 – Preparing the Dataset

df = pd.read_csv('./data/oscars.csv')

# Since we are most interested in awards related to 2023, let’s filter them and create a new Pandas dataframe.
df = df.loc[df['year_ceremony'] == 2023]
# also convert the category to lowercase while dropping the rows where the value of a film is blank.
df = df.dropna(subset=['film'])
df['category'] = df['category'].str.lower()
df.head()

# add a new column to the data frame that has an entire sentence representing a nomination.
# This complete sentence, when sent to GPT 3.5, enables it to find the facts within the context.

df['text'] = df['name'] + ' got nominated under the category, ' + \
    df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + \
    df['category'] + ', for the film ' + df['film'] + ' but did not win'

# Step 2 – Generate the Word Embeddings for the Dataset

# Set the embedding model to text-embedding-ada-002


def text_embedding(text) -> None:
    response = openai.Embedding.create(
        model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]


# use the text_embedding function to convert the query’s phrase or sentence into the same embedding format that Chorma uses.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)

client = chromadb.Client()
collection = client.get_or_create_collection(
    "oscars-2023", embedding_function=openai_ef)

# convert the text column in the Pandas dataframe into a Python list that can be passed to Chroma.
docs = df["text"].tolist()
# we will convert the index column of the dataframe into a list of strings.
# Since each document stored in Chroma also needs an id in the string format,
ids = [str(x) for x in df.index.tolist()]

collection.add(
    documents=docs,
    ids=ids
)

# Step 3 – Performing a Search to Retrieve Similar Text
# gets all the nominations for the music category.
vector = text_embedding("Nominations for music")
results = collection.query(
    query_embeddings=vector,
    n_results=15,
    include=["documents"]
)
# convert this list into one string that can provide context to the prompt.
res = "\n".join(str(item) for item in results['documents'][0])

prompt = f'```{res}```Based on the data in ```, answer who won the award for the original song'

messages = [
    {"role": "system", "content": "You answer questions about 95th Oscar awards."},
    {"role": "user", "content": prompt}
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0
)
response_message = response["choices"][0]["message"]["content"]

print(response_message)

# perform a cosine similarity search.
# Converts the query into embeddings and then compares it with each embedding available in the data frame.


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:

    query_embedding = text_embedding(query)

    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


strings, relatednesses = strings_ranked_by_relatedness(
    "Lady Gaga", df, top_n=3)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)

# One thing we want to make sure of is that the token size doesn’t exceed the supported context length of the model.
# For GPT 3.5, the context length is 4K.


def num_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# helper functions that make it easy to create the prompt by performing the similarity search in the data frame
# while respecting the token size.


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below content related to the 95th Oscar awards to answer the subsequent question. If the answer cannot be found in the content, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_row = f'\n\nOscar database section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_row + question)
            >
            token_budget
        ):
            break
        else:
            message += next_row
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about 95th Oscar awards."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


print(ask('What was the nomination from Lady Gaga for the 95th Oscars?'))
print(ask('What were the nominations for the music awards?'))
