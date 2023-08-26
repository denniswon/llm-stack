import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Generator, Optional
import json
import requests
from google.cloud import bigquery
import pandas as pd

from dotenv import load_dotenv

import requests
import json
import time

from sqlalchemy import create_engine
from typing import Optional

import multiprocessing as mp

load_dotenv()

etherscan_api_key = os.getenv('ETHERSCAN_API_KEY')
ABI_ENDPOINT = f'https://api.etherscan.io/api?module=contract&action=getabi&apikey={etherscan_api_key}&address='

conn_string = 'postgresql://postgres:postgres@127.0.0.1:5432/indexer'

db = create_engine(conn_string)
conn = db.connect()


def fetch_abi(address) -> Optional[str]:
    response = requests.get('%s%s' % (
        'https://api.etherscan.io/api?module=contract&action=getabi&address=', address))
    response_json = response.json()
    if (response_json['status'] != '1'):
        return None
    abi_json = json.loads(response_json['result'])
    return json.dumps({"abi": abi_json}, sort_keys=True)


def chunks(arr: list, n: int) -> Generator:
    """
    Yield successive n-sized chunks from arr.
    :param arr
    :param n
    :return generator
    """
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


if __name__ == '__main__':
    client = bigquery.Client()

    query = """
    SELECT
      address, bytecode
    FROM
      `bigquery-public-data.crypto_ethereum.contracts` AS contracts
    ORDER BY block_number DESC
    LIMIT 100
    """

    job_config = bigquery.job.QueryJobConfig(use_query_cache=True)
    query_job = client.query(query, job_config=job_config)

    iterator = query_job.result(timeout=30)
    rows = list(iterator)

    # Transform the rows into a nice pandas dataframe
    df = pd.DataFrame(data=[list(x.values())
                      for x in rows], columns=list(rows[0].keys()))

    print(f'Created: {query_job.created}')
    print(f'Ended:   {query_job.ended}')
    print(f'Bytes:   {query_job.total_bytes_processed:,}')

    address_chunks = chunks(df["address"].tolist(), 5)
    abis = []
    for addresses in address_chunks:
        pool = mp.Pool(mp.cpu_count())
        fetched = pool.map(fetch_abi, ((address) for address in addresses))
        pool.close()
        pool.join()
        abis += fetched
        time.sleep(1)

    print(len(abis))

    df['abi'] = abis
    df = df[df['abi'].isna() == False]
    print(df.head(10))

    df.to_sql('contracts', con=conn, if_exists='replace', index=False)
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()

    sql1 = '''select * from contracts;'''
    cursor.execute(sql1)
    for i in cursor.fetchall():
        print(i)

    # conn.commit()
    conn.close()
