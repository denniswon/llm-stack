import os
import pandas as pd
import openai

import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv

import requests
import json

from typing import Optional

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
ABI_ENDPOINT = 'https://api.etherscan.io/api?module=contract&action=getabi&address='

# Step 1 – Preparing the Dataset

df = pd.read_csv('./ethereumetl/contracts.csv')

# add a new column to the data frame that has an entire sentence representing a nomination.
# This complete sentence, when sent to GPT 3.5, enables it to find the facts within the context.

df['text'] = 'Smart contract with address ' + df['address'] + \
    ' has the bytecode of ' + df['bytecode']  # + '.' + \
# ('' if df['is_erc20'] == False & df['is_erc721'] == False else
#  ('\nthis contract is ' + 'erc20. ' if df['is_erc20'] == True else 'erc721. '))


def fetch_abi(address) -> Optional[str]:
    response = requests.get('%s%s' % (ABI_ENDPOINT, address))
    response_json = response.json()
    if (response_json['status'] != '1'):
        return None
    abi_json = json.loads(response_json['result'])
    return json.dumps({"abi": abi_json}, indent=4, sort_keys=True)


df = df.assign(abi=(df["address"].apply(lambda x: fetch_abi(x))))

df.loc[df['abi'] != None, 'text'] = df['text'] + \
    '\nThe contract abi is ' + df['abi']

print(df.head())

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

prompt = f'```{res}```Based on the data in ```, answer what is the abi of contract for the given address'

messages = [
    {"role": "system", "content": "You answer questions about abi of contract with address 0x6B175474E89094C44Da98b954EedeAC495271d0F"},
    {"role": "user", "content": prompt}
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0
)
response_message = response["choices"][0]["message"]["content"]

print(response_message)
